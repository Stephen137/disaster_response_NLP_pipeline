def detect_language(text):
    
    '''
    A function that leverages the langdetect library to detect the 
    source language of an input string and return the ISO 639-1 language codes
    '''
    
    '''
    INPUT:
    text - original message (str)
  
        
    OUTPUT:
    the languages dictionary key (str) in ISO 639-1 format
    '''        
    
    
    from langdetect import detect

    languages = {"af": "Afrikaans", 
                "ar": "Arabic",
                "bg": "Bulgarian",
                "bn": "Bengali",
                "ca": "Catalan",
                "cs": "Czech",
                "cy": "Welsh",
                "da": "Danish",
                "de": "German",
                "el": "Greek",
                "en": "English",
                "es": "Spanish",
                "et": "Estonian",
                "fa": "Persian",
                "fi": "Finnish",
                "fr": "French",
                "gu": "Gujarati",
                "he": "Hebrew",
                "hi": "Hindi",
                "hr": "Croatian",
                "hu": "Hungarian",
                "id": "Indonesian",
                "it": "Italian",
                "ja": "Japanese",
                "kn": "Kannada",
                "ko": "Korean",
                "lt": "Lithuanian",
                "lv": "Latvian",
                "mk": "Macedonian",
                "ml": "Malayalam",
                "mr": "Marathi",
                "ne": "Nepali",
                "nl": "Dutch",
                "no": "Norwegian",
                "pa": "Punjabi",
                "pl": "Polish",
                "pt": "Portuguese",
                "ro": "Romanian",
                "ru": "Russian",
                "sk": "Slovak",
                "sl": "Slovenian",
                "so": "Somali",
                "sq": "Albanian",
                "sv": "Swedish",
                "sw": "Swahili",
                "ta": "Tamili",
                "te": "Telugu",
                "th": "Thai",
                "tl": "Tagalog",
                "tr": "Turkish",
                "uk": "Ukrainian",
                "ur": "Urdu",
                "vi": "Vietnamese",
                "zh-cn": "Chinese - People's Republic of China",
                "zh-tw": "Chinese - Taiwan"  }

    if isinstance(text, str):
        
        try:
            language_code = detect(text)
            return languages.get(language_code, 'Unknown')
        except:
            return 'Unknown'
    else:
        return 'Unknown'

