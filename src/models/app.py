import os
import re
import sys

from flask import Flask, request, jsonify
from flask_cors import CORS
import concurrent.futures
import mwapi
import requests
import time
import torch
from transformers import MBartTokenizer
from transformers import BertModel, BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers import MBartConfig
import yaml

from . import MBartForConditionalGenerationDescartes

BERT_PATH = "bert-base-multilingual-uncased"
LANG_DICT = {"en": "en_XX", "fr": "fr_XX", "it": "it_IT", "es": "es_XX", "de": "de_DE", "nl": "nl_XX", "ja": "ja_XX",
             "zh": "zh_CN", "ko": "ko_KR", "vi": "vi_VN", "ru": "ru_RU", "cs": "cs_CZ", "fi": "fi_FI", "lt": "lt_LT",
             "lv": "lv_LV", "et": "et_EE", "ar": "ar_AR", "tr": "tr_TR", "ro": "ro_RO", "kk": "kk_KZ", "gu": "gu_IN",
             "hi": "hi_IN", "si": "si_LK", "my": "my_MM", "ne": "ne_NP"}
DATE_REGEX = re.compile('\d{4}')

def prepare_inputs(inputs, device):
    """
    Prepare :obj:`inputs` before feeding them to the model, converting them to tensors if they are not already and
    handling potential state.
    """
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)
        elif isinstance(v, dict):
            for key, val in v.items():
                if isinstance(val, torch.Tensor):
                    v[key] = val.to(device)
                elif isinstance(val, BatchEncoding) or isinstance(val, dict):
                    for k1, v1 in val.items():
                        if isinstance(v1, torch.Tensor):
                            val[k1] = v1.to(device)
    return inputs

class ModelLoader:

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.tokenizer_bert = None
        self.device = None

    def load_model(self, output_dir):
        config = MBartConfig.from_pretrained(output_dir)
        config.graph_embd_length = 128  # not used but still expected
        model = MBartForConditionalGenerationDescartes.from_pretrained(output_dir, config=config)
        tokenizer = MBartTokenizer.from_pretrained(output_dir)

        tokenizer_bert = BertTokenizer.from_pretrained(BERT_PATH)
        bert_model = BertModel.from_pretrained(BERT_PATH)
        model.model_bert = bert_model

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_bert = tokenizer_bert
        self.device = device

    def predict(self, sources, descriptions, tgt_lang, num_beams=1, num_return_sequences=1):
        batch = {}
        input_ids = {}
        attention_mask = {}
        # process first paragraphs
        for lang, lang_code in LANG_DICT.items():
            if lang in sources:
                source = sources[lang]
            else:
                source = ""
            if len(source) > 0:
                self.tokenizer.src_lang = lang_code
                batch_enc = self.tokenizer([source], padding=True, truncation=True)
                input_ids[lang] = torch.tensor(batch_enc['input_ids'])
                attention_mask[lang] = torch.tensor(batch_enc['attention_mask'])
            else:
                input_ids[lang] = None
                attention_mask[lang] = None

        # process descriptions
        bert_inputs = {}
        for lang, description in descriptions:
            if lang != tgt_lang:
                bert_outs = self.tokenizer_bert([description],
                                                padding=True,
                                                truncation=True,
                                                return_tensors="pt")
                bert_inputs[lang] = bert_outs

        batch['input_ids'] = input_ids
        batch['attention_mask'] = attention_mask
        batch["graph_embeddings"] = None
        batch['bert_inputs'] = bert_inputs

        batch = prepare_inputs(batch, self.device)
        tokens = self.model.generate(**batch, max_length=20, min_length=2, length_penalty=2.0, num_beams=num_beams,
                                     early_stopping=True, target_lang=LANG_DICT[tgt_lang],
                                     decoder_start_token_id=self.tokenizer.lang_code_to_id[LANG_DICT[tgt_lang]],
                                     num_return_sequences=num_return_sequences)
        output = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        return output

__dir__ = os.path.dirname(__file__)
__updir = os.path.abspath(os.path.join(__dir__, '..'))
sys.path.append(__updir)

app = Flask(__name__)

MODEL = ModelLoader()
SUPPORTED_WIKIPEDIA_LANGUAGE_CODES = ['en', 'de', 'nl', 'es', 'it', 'ru', 'fr', 'zh', 'ar', 'vi', 'ja', 'fi', 'ko',
                                      'tr', 'ro', 'cs', 'et', 'lt', 'kk', 'lv', 'hi', 'ne', 'my', 'si', 'gu']
LANGS_BY_SIZE = {l:i for i,l in enumerate(SUPPORTED_WIKIPEDIA_LANGUAGE_CODES)}

# load in app user-agent or any other app config
app.config.update(
    yaml.safe_load(open(os.path.join(__updir, 'flask_config.yaml'))))

# Enable CORS for API endpoints
cors = CORS(app, resources={r'/article': {'origins': '*'},
                            r'/supported-languages': {'origins': '*'}})


@app.route('/supported-languages', methods=['GET'])
def get_supported_languages():
    return jsonify({'languages': SUPPORTED_WIKIPEDIA_LANGUAGE_CODES})


@app.route('/article', methods=['GET'])
def get_article_description():
    lang, title, num_beams, num_langs, error = validate_api_args()
    if error:
        return jsonify({'error': error})
    else:
        return jsonify(run_model(lang, title, num_beams, num_langs))


def run_model(lang, title, num_beams, num_langs=25):
    execution_times = {}  # just used right now for debugging
    features = {}  # just used right now for debugging
    starttime = time.time()

    dates = set()
    descriptions, sitelinks, blp = get_wikidata_info(lang, title)
    wd_time = time.time()
    execution_times['wikidata-info (s)'] = wd_time - starttime
    features['descriptions'] = descriptions
    features['sitelinks'] = list(sitelinks.keys())
    sitelinks = {l: sitelinks[l] for i, l in enumerate(sorted(sitelinks, key=LANGS_BY_SIZE.get)) if i <= num_langs}

    first_paragraphs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = { executor.submit(get_first_paragraph, l, sitelinks[l]): l for l in sitelinks }
        futures[executor.submit(get_groundtruth, lang, title)] = 'groundtruth'
        for future in concurrent.futures.as_completed(futures):
            if futures[future] == 'groundtruth':
                groundtruth_desc = future.result()
            else:
                first_paragraphs[futures[future]] = future.result()

    execution_times['total network (s)'] = time.time() - starttime
    features['first-paragraphs'] = first_paragraphs

    prediction = MODEL.predict(first_paragraphs, descriptions, lang,
                               num_beams=num_beams, num_return_sequences=num_beams)
    execution_times['model (s)'] = time.time() - starttime - execution_times['total network (s)']

    for d in descriptions:
        dates.update(DATE_REGEX.findall(d))
    for p in first_paragraphs:
        dates.update(DATE_REGEX.findall(p))
    filtered_predictions = []
    for p in prediction:
        dates_in_p = DATE_REGEX.findall(p)
        keep = True
        for d in dates_in_p:
            if d not in dates:
                keep = False
                break
        if keep:
            filtered_predictions.append(p)

    execution_times['total (s)'] = time.time() - starttime

    return {'lang': lang, 'title': title, 'blp':blp,
            'num_beams':num_beams,
            'groundtruth': groundtruth_desc,
            'latency': execution_times,
            'features': features,
            'prediction':prediction}


def get_first_paragraph(lang, title):
    # get plain-text extract of article
    try:
        response = requests.get(f'https://{lang}.wikipedia.org/api/rest_v1/page/summary/{title}', headers={ 'User-Agent': app.config['CUSTOM_UA'] })
        return response.json()['extract']
    except Exception:
        return ''

def get_groundtruth(lang, title):
    """Get existing article description (groundtruth)."""
    session = mwapi.Session(f'https://{lang}.wikipedia.org', user_agent=app.config['CUSTOM_UA'])

    # English has a prop that takes into account shortdescs (local override) that other languages don't
    if lang == 'en':
        result = session.get(
            action="query",
            prop="pageprops",
            titles=title,
            redirects="",
            format='json',
            formatversion=2
        )
        try:
            return result['query']['pages'][0]['pageprops']['wikibase-shortdesc']
        except Exception:
            return None
    # Non-English languages: get description from Wikidata
    else:
        # https://fr.wikipedia.org/w/api.php?action=query&prop=pageterms&titles=Chicago&wbptterms=description&wbptlanguage=fr&format=json&formatversion=2
        result = session.get(
            action="query",
            prop="pageterms",
            titles=title,
            redirects="",
            wbptterms="description",
            wbptlanguage=lang,
            format='json',
            formatversion=2
        )
        try:
            return result['query']['pages'][0]['terms']['description'][0]
        except Exception:
            return None

def get_wikidata_info(lang, title):
    """Get article descriptions from Wikidata"""
    session = mwapi.Session('https://wikidata.org', user_agent=app.config['CUSTOM_UA'])

    result = session.get(
        action="wbgetentities",
        sites=f"{lang}wiki",
        titles=title,
        redirects="yes",
        props='descriptions|claims|sitelinks',
        languages="|".join(SUPPORTED_WIKIPEDIA_LANGUAGE_CODES),
        sitefilter="|".join([f'{l}wiki' for l in SUPPORTED_WIKIPEDIA_LANGUAGE_CODES]),
        format='json',
        formatversion=2
    )

    descriptions = {}
    sitelinks = {}
    blp = False
    try:
        # should be exactly 1 QID for the page if it has a Wikidata item
        qid = list(result['entities'].keys())[0]
        # get all the available descriptions in relevant languages
        for l in result['entities'][qid]['descriptions']:
            descriptions[l] = result['entities'][qid]['descriptions'][l]['value']
        # get the sitelinks from supported languages
        for wiki in result['entities'][qid]['sitelinks']:
            lang = wiki[:-4]  # remove 'wiki' part
            sitelinks[lang] = result['entities'][qid]['sitelinks'][wiki]['title']
        try:
            human = False
            claims = result['entities'][qid]['claims']
            for io_claim in claims.get('P31', []):
                if io_claim['mainsnak']['datavalue']['value']['id'] == 'Q5':
                    human = True
                    break
            died = 'P570' in claims  # date-of-death property
            if human and not died:
                blp = True
        except Exception:
            pass  # ok to error out on this and keep rest of info -- that says likely not BLP
    except Exception:
        pass

    return descriptions, sitelinks, blp

def get_canonical_page_title(title, lang):
    """Resolve redirects / normalization -- used to verify that an input page_title exists and help future API calls"""
    session = mwapi.Session('https://{0}.wikipedia.org'.format(lang), user_agent=app.config['CUSTOM_UA'])

    result = session.get(
        action="query",
        prop="info",
        inprop='',
        redirects='',
        titles=title,
        format='json',
        formatversion=2
    )
    if 'missing' in result['query']['pages'][0]:
        return None
    else:
        return result['query']['pages'][0]['title'].replace(' ', '_')

def validate_lang(lang):
    return lang in SUPPORTED_WIKIPEDIA_LANGUAGE_CODES

def validate_api_args():
    """Validate API arguments: supported Wikipedia language and valid page title."""
    error = None
    lang = None
    page_title = None
    if request.args.get('title') and request.args.get('lang'):
        lang = request.args['lang']
        page_title = get_canonical_page_title(request.args['title'], lang)
        if page_title is None:
            error = 'no matching article for <a href="https://{0}.wikipedia.org/wiki/{1}">https://{0}.wikipedia.org/wiki/{1}</a>'.format(lang, request.args['title'])
    elif request.args.get('lang'):
        error = 'missing an article title -- e.g., "2005_World_Series" for <a href="https://en.wikipedia.org/wiki/2005_World_Series">https://en.wikipedia.org/wiki/2005_World_Series</a>'
    elif request.args.get('title'):
        error = 'missing a language -- e.g., "en" for English'
    else:
        error = 'missing language -- e.g., "en" for English -- and title -- e.g., "2005_World_Series" for <a href="https://en.wikipedia.org/wiki/2005_World_Series">https://en.wikipedia.org/wiki/2005_World_Series</a>'

    num_beams = 1
    if request.args.get('num_beams'):
        try:
            num_beams = max(int(request.args['num_beams']), num_beams)  # must return at least one sequence
        except Exception:
            pass

    num_langs = 25
    if request.args.get('num_langs'):
        try:
            num_langs = int(request.args['num_langs'])  # must return at least one sequence
        except Exception:
            pass

    return lang, page_title, num_beams, num_langs, error


def load_model():
    # Load model (takes ~1 minute) and prime with first prediction
    # to make sure operating correctly and fully loaded in
    model_path = '/Users/ijohnson/Downloads/model-25lang-notype'
    MODEL.load_model(model_path)
    test_model()

def test_model():
    lang = 'en'
    title = 'Clandonald'
    num_beams = 2

    expected = ['Hamlet in Alberta, Canada', 'human settlement in Alberta, Canada']
    result = run_model(lang, title, num_beams)
    result['expected'] = expected
    print(result)

load_model()
application = app

if __name__ == '__main__':
    application.run()