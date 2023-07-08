# -*- coding: utf-8 -*-
import re
import ast
import sys
import token
import tokenize
import ast
import re
import ast
import io
import logging
import tokenize
from nltk.tokenize import wordpunct_tokenize
import re
from nltk.corpus import wordnet
from nltk import wordpunct_tokenize
from io import StringIO
import re
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
# 骆驼命名法
import inflection

# 词性还原
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
wnler = WordNetLemmatizer()

# 词干提取
from nltk.corpus import wordnet

#############################################################################
import re

FLAG_IN = 1
FLAG_OUT = 2
FLAG_CONT = 3
FLAG_PROMPT_IN = 4
FLAG_PROMPT_CONT = 5


def repair_program_io(code):
    # regex patterns for case 1
    pattern_case1_in = re.compile(r"In ?\[\d+\]: ?")
    pattern_case1_out = re.compile(r"Out ?\[\d+\]: ?")
    pattern_case1_cont = re.compile(r"( )+\.+: ?")

    # regex patterns for case 2
    pattern_case2_in = re.compile(r">>> ?")
    pattern_case2_cont = re.compile(r"\.\.\. ?")

    regex_patterns = [pattern_case1_in, pattern_case1_out, pattern_case1_cont,
                      pattern_case2_in, pattern_case2_cont]

    code_lines = code.split("\n")
    line_flags = [0] * len(code_lines)
    code_blocks = []

    # match patterns
    for idx, line in enumerate(code_lines):
        for flag_idx, pattern in enumerate(regex_patterns):
            if re.match(pattern, line):
                line_flags[idx] = flag_idx + 1
                break

    # check if repair is needed
    if all(flag == 0 for flag in line_flags):
        return code, [code]

    # repair
    repaired_code = ""
    current_block = ""
    is_repaired = False

    for idx, line in enumerate(code_lines):
        if line_flags[idx] == 0:
            repaired_code += line + "\n"
            if current_block:
                code_blocks.append(current_block.strip())
                current_block = ""
        elif line_flags[idx] in [FLAG_IN, FLAG_CONT, FLAG_PROMPT_CONT]:
            current_block += re.sub(regex_patterns[line_flags[idx] - 1], "", line) + "\n"
        elif line_flags[idx] == FLAG_OUT:
            current_block = re.sub(regex_patterns[line_flags[idx] - 1], "", line) + "\n"
        elif line_flags[idx] == FLAG_PROMPT_IN:
            current_block = ""

        if line_flags[idx] in [FLAG_OUT, FLAG_CONT]:
            is_repaired = True

    if current_block:
        code_blocks.append(current_block.strip())

    if not is_repaired:
        return code, [code]

    return repaired_code.strip(), code_blocks


PATTERN_VAR_EQUAL = re.compile(r"(\s*[_a-zA-Z][_a-zA-Z0-9]*\s*)(,\s*[_a-zA-Z][_a-zA-Z0-9]*\s*)*=")
PATTERN_VAR_FOR = re.compile(r"for\s+[_a-zA-Z][_a-zA-Z0-9]*\s*(,\s*[_a-zA-Z][_a-zA-Z0-9]*)*\s+in")


def get_variable_names(ast_root):
    return sorted({node.id for node in ast.walk(ast_root) if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load)})


def get_variable_names_heuristics(code):
    variable_names = set()
    code_lines = [_ for _ in code.split("\n") if len(_.strip())]

    # best effort parsing
    start = 0
    end = len(code_lines) - 1
    is_parsed_successfully = False
    while not is_parsed_successfully:
        try:
            ast_root = ast.parse("\n".join(code_lines[start:end]))
        except:
            end -= 1
        else:
            is_parsed_successfully = True
    variable_names = variable_names.union(set(get_variable_names(ast_root)))

    # processing the remaining...
    for line in code_lines[end:]:
        line = line.strip()
        try:
            ast_root = ast.parse(line)
        except:
            if PATTERN_VAR_EQUAL.match(line):
                match = PATTERN_VAR_EQUAL.match(line).group()[:-1]  # remove "="
                variable_names = variable_names.union({_.strip() for _ in match.split(",")})

            elif PATTERN_VAR_FOR.search(line):
                match = PATTERN_VAR_FOR.search(line).group()[3:-2]  # remove "for" and "in"
                variable_names = variable_names.union({_.strip() for _ in match.split(",")})

            else:
                continue
        else:
            variable_names = variable_names.union(get_variable_names(ast_root))

    return variable_names

logger = logging.getLogger(__name__)


def PythonParser(code):
    try:
        root = ast.parse(code)
        varnames = set(get_vars(root))
    except Exception:
        repaired_code, _ = repair_program_io(code)
        try:
            root = ast.parse(repaired_code)
            varnames = set(get_vars(root))
        except Exception:
            bool_failed_var = True
            varnames = get_vars_heuristics(code)

    tokenized_code = []

    def first_trial(_code):
        if len(_code) == 0:
            return True
        try:
            g = tokenize.generate_tokens(io.StringIO(_code).readline)
            term = next(g)
        except StopIteration:
            return False
        else:
            return True

    bool_first_success = first_trial(code)
    while not bool_first_success:
        code = code[1:]
        bool_first_success = first_trial(code)
    g = tokenize.generate_tokens(io.StringIO(code).readline)
    term = next(g)

    bool_finished = False
    while not bool_finished:
        term_type = term[0]
        lineno = term[2][0] - 1
        posno = term[3][1] - 1
        if token.tok_name[term_type] in {"NUMBER", "STRING", "NEWLINE"}:
            tokenized_code.append(token.tok_name[term_type])
        elif token.tok_name[term_type] not in {"COMMENT", "ENDMARKER"} and len(term[1].strip()):
            candidate = term[1].strip()
            if candidate not in varnames:
                tokenized_code.append(candidate)
            else:
                tokenized_code.append("VAR")

        # fetch the next term
        bool_success_next = False
        while not bool_success_next:
            try:
                term = next(g)
            except StopIteration:
                bool_finished = True
                break
            except Exception as e:
                logger.exception(f"Failed to tokenize line {lineno}: {e}")
                # tokenize the error line with wordpunct_tokenizer
                code_lines = code.split("\n")
                if lineno <= len(code_lines) - 1:
                    failed_code_line = code_lines[lineno]  # error line
                    if posno < len(failed_code_line) - 1:
                        failed_code_line = failed_code_line[posno:]
                        tokenized_failed_code_line = wordpunct_tokenize(
                            failed_code_line)  # tokenize the failed line segment
                        tokenized_code += tokenized_failed_code_line
                    if lineno < len(code_lines) - 1:
                        code = "\n".join(code_lines[lineno + 1:])
                        g = tokenize.generate_tokens(io.StringIO(code).readline)
                    else:
                        bool_finished = True
                        break
            else:
                bool_success_next = True

    return tokenized_code, bool_failed_var, bool_failed_token

#############################################################################

#缩略词处理
PAT_IS = re.compile("(it|he|she|that|this|there|here)(\"s)", re.I)
PAT_S1 = re.compile("(?<=[a-zA-Z])\"s")
PAT_S2 = re.compile("(?<=s)\"s?")
PAT_NOT = re.compile("(?<=[a-zA-Z])n\"t")
PAT_WOULD = re.compile("(?<=[a-zA-Z])\"d")
PAT_WILL = re.compile("(?<=[a-zA-Z])\"ll")
PAT_AM = re.compile("(?<=[I|i])\"m")
PAT_ARE = re.compile("(?<=[a-zA-Z])\"re")
PAT_VE = re.compile("(?<=[a-zA-Z])\"ve")


def revert_abbrev(line):
    line = PAT_IS.sub(r"\1 is", line)
    line = PAT_S1.sub("", line)
    line = PAT_S2.sub("", line)
    line = PAT_NOT.sub(" not", line)
    line = PAT_WOULD.sub(" would", line)
    line = PAT_WILL.sub(" will", line)
    line = PAT_AM.sub(" am", line)
    line = PAT_ARE.sub(" are", line)
    line = PAT_VE.sub(" have", line)
    return line

#获取词性
def get_wordnet_pos(tag):
    """
    Map POS tag to WordNet POS tag.

    Parameters:
        tag (str): The POS tag.

    Returns:
        One of the WordNet POS constants: wordnet.ADJ, wordnet.NOUN,
        wordnet.VERB, wordnet.ADV, or None.
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None



# ---------------------子函数1：句子的去冗与分词--------------------


PAT_DECIMAL = re.compile(r"\d+(\.\d+)+")
PAT_STRING = re.compile(r'\"[^\"]+\"')
PAT_HEX = re.compile(r"0[xX][A-Fa-f0-9]+")
PAT_NUMBER = re.compile(r"\s?\d+\s?")
PAT_OTHER = re.compile(r"(?<![A-Z|a-z|_|])\d+[A-Za-z]+")

wnler = WordNetLemmatizer()


def process_nl_line(line):
    line = revert_abbrev(line)
    line = re.sub(r'[\t\n]+', ' ', line).strip()
    line = inflection.underscore(line)
    line = re.sub(r"\([^\(|^\)]+\)", '', line).strip()
    return line


def process_sent_word(line):
    line = re.sub(PAT_DECIMAL, 'TAGINT', line)
    line = re.sub(PAT_STRING, 'TAGSTR', line)
    line = re.sub(PAT_HEX, 'TAGINT', line)
    line = re.sub(PAT_NUMBER, ' TAGINT ', line)
    line = re.sub(PAT_OTHER, 'TAGOER', line)
    words = word_tokenize(line.lower())
    word_tags = pos_tag(words)
    word_list = []
    for word, tag in word_tags:
        word_pos = get_wordnet_pos(tag)
        if word_pos in ['a', 'v', 'n', 'r']:
            word = wnler.lemmatize(word, pos=word_pos)
        word = wordnet.morphy(word) or word
        word_list.append(word)
    return word_list


def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


#############################################################################


PAT_INVACHAR_ALL = re.compile('[^(0-9|a-z|A-Z|\-|_|\'|\"|\-|\(|\)|\n)]+')
PAT_INVACHAR_PART = re.compile('[^(0-9|a-z|A-Z|\-|#|/|_|,|\'|=|>|<|\"|\-|\\|\(|\)|\?|\.|\*|\+|\[|\]|\^|\{|\}|\n)]+')

def filter_all_invachar(line):
    line = re.sub(PAT_INVACHAR_ALL, ' ', line)
    line = re.sub('-+', '-', line)
    line = re.sub('_+', '_', line)
    line = line.replace('|', ' ').replace('¦', ' ')
    return line


def filter_part_invachar(line):
    line = re.sub(PAT_INVACHAR_PART, ' ', line)
    line = re.sub('-+', '-', line)
    line = re.sub('_+', '_', line)
    line = line.replace('|', ' ').replace('¦', ' ')
    return line


def python_code_parse(line):
    if not isinstance(line, str):
        raise TypeError("Input line must be a string")
    if len(line) == 0:
        raise ValueError("Input line cannot be empty")

    line = filter_part_invachar(line)
    line = re.sub('[\t\n]+', ' ', line)
    line = re.sub('\.+', '.', line)
    line = re.sub(' +', ' ', line)
    line = line.strip()

    try:
        typedCode, failed_var, failed_token = PythonParser(line)
        typedCode = inflection.underscore(' '.join(typedCode)).split(' ')
        cut_tokens = [re.sub("\s+", " ", x.strip()) for x in typedCode]
        token_list = [x.lower() for x in cut_tokens if x.strip()]
        return token_list
    except Exception as e:
        print("Error: ", e)
        return None


########################主函数：代码的tokens#################################


#######################主函数：句子的tokens##################################
PAT_INVACHAR_ALL = re.compile('[^(0-9|a-z|A-Z|\-|_|\'|\"|\-|\(|\)|\n)]+')
PAT_INVACHAR_PART = re.compile('[^(0-9|a-z|A-Z|\-|#|/|_|,|\'|=|>|<|\"|\-|\\|\(|\)|\?|\.|\*|\+|\[|\]|\^|\{|\}|\n)]+')

def filter_all_invachar(line):
    line = re.sub(PAT_INVACHAR_ALL, ' ', line)
    line = re.sub('-+', '-', line)
    line = re.sub('_+', '_', line)
    line = line.replace('|', ' ').replace('¦', ' ')
    return line


def filter_part_invachar(line):
    line = re.sub(PAT_INVACHAR_PART, ' ', line)
    line = re.sub('-+', '-', line)
    line = re.sub('_+', '_', line)
    line = line.replace('|', ' ').replace('¦', ' ')
    return line


def process_parenthesis(word_list):
    for i in range(len(word_list)):
        if re.findall('[\(\)]', word_list[i]):
            word_list[i] = ''
    return word_list


def python_query_parse(line):
    if not isinstance(line, str):
        raise TypeError("Input line must be a string")
    if len(line) == 0:
        raise ValueError("Input line cannot be empty")

    line = filter_all_invachar(line)
    line = process_nl_line(line)
    word_list = process_sent_word(line)
    word_list = process_parenthesis(word_list)
    word_list = [w.strip() for w in word_list if w.strip()]
    return word_list


def python_context_parse(line):
    if not isinstance(line, str):
        raise TypeError("Input line must be a string")
    if len(line) == 0:
        raise ValueError("Input line cannot be empty")

    line = filter_part_invachar(line)
    line = process_nl_line(line)
    word_list = process_sent_word(line)
    word_list = [w.strip() for w in word_list if w.strip()]
    return word_list


#######################主函数：句子的tokens##################################

if __name__ == '__main__':

    print(python_query_parse("change row_height and column_width in libreoffice calc use python tagint"))
    print(python_query_parse('What is the standard way to add N seconds to datetime.time in Python?'))
    print(python_query_parse("Convert INT to VARCHAR SQL 11?"))
    print(python_query_parse('python construct a dictionary {0: [0, 0, 0], 1: [0, 0, 1], 2: [0, 0, 2], 3: [0, 0, 3], ...,999: [9, 9, 9]}'))

    print(python_context_parse('How to calculateAnd the value of the sum of squares defined as \n 1^2 + 2^2 + 3^2 + ... +n2 until a user specified sum has been reached sql()'))
    print(python_context_parse('how do i display records (containing specific) information in sql() 11?'))
    print(python_context_parse('Convert INT to VARCHAR SQL 11?'))

    print(python_code_parse('if(dr.HasRows)\n{\n // ....\n}\nelse\n{\n MessageBox.Show("ReservationAnd Number Does Not Exist","Error", MessageBoxButtons.OK, MessageBoxIcon.Asterisk);\n}'))
    print(python_code_parse('root -> 0.0 \n while root_ * root < n: \n root = root + 1 \n print(root * root)'))
    print(python_code_parse('root = 0.0 \n while root * root < n: \n print(root * root) \n root = root + 1'))
    print(python_code_parse('n = 1 \n while n <= 100: \n n = n + 1 \n if n > 10: \n  break print(n)'))
    print(python_code_parse("diayong(2) def sina_download(url, output_dir='.', merge=True, info_only=False, **kwargs):\n    if 'news.sina.com.cn/zxt' in url:\n        sina_zxt(url, output_dir=output_dir, merge=merge, info_only=info_only, **kwargs)\n  return\n\n    vid = match1(url, r'vid=(\\d+)')\n    if vid is None:\n        video_page = get_content(url)\n        vid = hd_vid = match1(video_page, r'hd_vid\\s*:\\s*\\'([^\\']+)\\'')\n  if hd_vid == '0':\n            vids = match1(video_page, r'[^\\w]vid\\s*:\\s*\\'([^\\']+)\\'').split('|')\n            vid = vids[-1]\n\n    if vid is None:\n        vid = match1(video_page, r'vid:\"?(\\d+)\"?')\n    if vid:\n   sina_download_by_vid(vid, output_dir=output_dir, merge=merge, info_only=info_only)\n    else:\n        vkey = match1(video_page, r'vkey\\s*:\\s*\"([^\"]+)\"')\n        if vkey is None:\n            vid = match1(url, r'#(\\d+)')\n            sina_download_by_vid(vid, output_dir=output_dir, merge=merge, info_only=info_only)\n            return\n        title = match1(video_page, r'title\\s*:\\s*\"([^\"]+)\"')\n        sina_download_by_vkey(vkey, title=title, output_dir=output_dir, merge=merge, info_only=info_only)"))

    print(python_code_parse("d = {'x': 1, 'y': 2, 'z': 3} \n for key in d: \n  print (key, 'corresponds to', d[key])"))
    print(python_code_parse('  #       page  hour  count\n # 0     3727441     1   2003\n # 1     3727441     2    654\n # 2     3727441     3   5434\n # 3     3727458     1    326\n # 4     3727458     2   2348\n # 5     3727458     3   4040\n # 6   3727458_1     4    374\n # 7   3727458_1     5   2917\n # 8   3727458_1     6   3937\n # 9     3735634     1   1957\n # 10    3735634     2   2398\n # 11    3735634     3   2812\n # 12    3768433     1    499\n # 13    3768433     2   4924\n # 14    3768433     3   5460\n # 15  3768433_1     4   1710\n # 16  3768433_1     5   3877\n # 17  3768433_1     6   1912\n # 18  3768433_2     7   1367\n # 19  3768433_2     8   1626\n # 20  3768433_2     9   4750\n'))
