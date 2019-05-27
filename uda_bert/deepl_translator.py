from requests import Session
from argparse import ArgumentParser

json = {"jsonrpc": "2.0",
        "method": "LMT_handle_jobs",
        "params": {
            "jobs": [],
            "lang": {},
            "priority": 1
        }
        }

part = {
    "kind": "default"
}


class DeeplTranslator:
    def __init__(self, json=json, part=part, url='https://www2.deepl.com/jsonrpc'):
        self._session = Session()
        self._json = json
        self._part = part
        self._url = url

    def translate(self, text, selected_lang, target_lang):
        self._part['raw_en_sentence'] = text
        self._json.get('params').get('jobs').append(self._part)

        lang = self._json.get('params').get('lang')
        lang['source_lang_user_selected'] = selected_lang
        lang['target_lang'] = target_lang

        response = self._session.post(url=self._url,
                                      json=self._json)
        return response.json().get('result'), response.json().get('result').get('translations')[0].get('beams')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--text", "-d", type=str, required=True, help="Text to translate")
    parser.add_argument("--selected_lang", "-s", type=str, default='EN', help="Source language")
    parser.add_argument("--target_lang", "-t", type=str, default='FR', help="Target language")

    args = parser.parse_args()
    text = args.text
    selected_lang = args.selected_lang
    target_lang = args.target_lang

    translator = DeeplTranslator()
    full_res, translations = translator.translate(text, selected_lang, target_lang)
    # print(full_res)
    print(translations)
