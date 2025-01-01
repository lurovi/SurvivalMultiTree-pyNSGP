def only_first_char_upper(s: str) -> str:
    return s[0].upper() + s[1:]


def concat(s1: str, s2: str, sep: str = '') -> str:
    return s1 + sep + s2


def multiple_concat(s: list[str], sep: str = '') -> str:
    res: str = ''
    for i in range(len(s)):
        ss: str = s[i]
        res = res + sep + ss if i != 0 else res + ss
    return res


def extract_digits(s: str) -> str:
    res: str = ''
    for c in s:
        if c.isdigit():
            res += c
    return res


def is_vowel(c: str) -> bool:
    return c.upper() in ('A', 'E', 'I', 'O', 'U')


def is_consonant(c: str) -> bool:
    return c.isalpha() and c.upper() not in ('A', 'E', 'I', 'O', 'U')


def acronym(s: str, n_chars: int = 3) -> str:
    u: str = s.upper()
    digits: str = extract_digits(u)
    if len(digits) >= n_chars:
        raise ValueError(f'{n_chars} is the number of characters of the acronym, however {len(digits)} is the number of digits in the string, hence no alphabetic character can appear in the acronym, please either increase the number of characters in the acronym or get rid of the digits in the string.')
    acronym_size: int = n_chars - len(digits)
    res: str = '' + u[0]
    count: int = 1
    for i in range(1, len(u)):
        c = u[i]
        if count == acronym_size:
            break
        if is_consonant(c):
            res += c
            count += 1
    res = res + digits
    return res

