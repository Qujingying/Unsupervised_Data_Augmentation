import requests
from argparse import ArgumentParser

url = "https://www.bing.com/ttranslate"

headers = {
    'cache-control': "no-cache"
}


def bing_translator(text, selected_lang, target_lang):
    params = {'text': text,
              'from': selected_lang,
              'to': target_lang}
    response = requests.request("POST", url, headers=headers, params=params)

    return response.json()["translationResponse"]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--text", "-d", type=str, required=True, help="Text to translate")
    parser.add_argument("--selected_lang", "-s", type=str, default='en', help="Source language")
    parser.add_argument("--target_lang", "-t", type=str, default='fr', help="Target language")

    args = parser.parse_args()
    text = args.text
    selected_lang = args.selected_lang
    target_lang = args.target_lang

    translation = bing_translator(text, selected_lang, target_lang)
    print(translation)
