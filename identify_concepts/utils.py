import regex as re

def custom_cleaning_rules(objects):
    """
    objects can be a List[str] or str
    """
    
    # Clean non-text characters that tend to occur 
    re_patterns = [r'\d+\s?\d+', r'[\\\(\)]']
    
    input_type = 'list'
    if type(objects) == str:
        input_type = 'str'
        objects = [objects]

    cleaned_objects = []
    for obj in objects:
        # remove double determiners that are sometimes grabbed, and strip objects
        obj = obj.replace("thethe", '', 1).strip()
        obj = obj.replace("thenthe", '', 1).strip()
        obj = obj.replace("thethat", '', 1).strip()
        obj = obj.replace("their ", '', 1).strip()
        obj = obj.replace(". ", '').strip()

        for p in re_patterns:
            obj = re.sub(p, '', obj)
        
        if len(obj) == 1:
            # remove 1 character objects
            continue
        elif len(obj) < 4 and (not obj.isupper() or any(c for c in obj if (c.isdigit() or c.isspace()))):
            # remove 2 & 3 characters objects that aren't all uppercase (abbreviations?) / contain a number or space
            # while removing some 3 letter words like 'ice' and 'fan', most of these are uninformative/erroneous
            continue
        elif len(obj) < 6 and len(re.findall(r"[^\w\s]", obj)) > 1:
            # any span of 5 characters or less, that contains multiple non-word and non-space characters
            continue
        elif len(re.findall(r"[=+*@|<>»_%]", obj)) > 0:
            # any span that may indicate that its taken from an equation or email address or simply gibberish from ocr
            continue
        elif obj.startswith("the ") or obj.startswith("The ") or obj.startswith("a ") or obj.startswith("A "):
            # do the same 1 char and 2/3 char removal in case the object starts with a determiner;
            if len(obj) == 5:
                continue
            elif len(obj) < 8 and obj[4:].islower():
                continue
            else:
                cleaned_objects.append(obj)
        else:
            cleaned_objects.append(obj)
     
    if input_type == 'list':
        return list(set(cleaned_objects))
    if input_type == 'str':
        try:
            return cleaned_objects[0]
        except IndexError:
            return ''