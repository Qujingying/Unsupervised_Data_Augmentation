from argparse import ArgumentParser
from googletrans import Translator

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--text", "-d", type=str, required=True, help="Text to translate")
    parser.add_argument("--selected_lang", "-s", type=str, default='en', help="Source language")
    parser.add_argument("--target_lang", "-t", type=str, default='fr', help="Target language")

    args = parser.parse_args()
    text = args.text
    selected_lang = args.selected_lang
    target_lang = args.target_lang

    translator = Translator()
    translation = translator.translate(text, src=selected_lang, dest=target_lang)
    print(translation.text)
    # We can set an array of text : ['text1', 'text2', ...] and we get an iterator to be process like this
    # for translation in translations:
    #     print(translation.text)
