# -*- coding: utf-8 -*-
import re
import sqlparse #0.4.2
from nltk.corpus import wordnet
from nltk import pos_tag
import inflection
from nltk.stem import WordNetLemmatizer
wnler = WordNetLemmatizer()

#############################################################################
class SqlangParser:
    ttypes = {0: "OTHER", 1: "FUNCTION", 2: "BLANK", 3: "KEYWORD", 4: "INTERNAL", 5: "TABLE", 6: "COLUMN", 7: "INTEGER",
              8: "FLOAT", 9: "HEX", 10: "STRING", 11: "WILDCARD", 12: "SUBQUERY", 13: "DUD"}

    scanner = re.Scanner(
        [(r"\[[^\]]*\]", lambda scanner, token: token), (r"\+", lambda scanner, token: "REGPLU"),
         (r"\*", lambda scanner, token: "REGAST"), (r"%", lambda scanner, token: "REGCOL"),
         (r"\^", lambda scanner, token: "REGSTA"), (r"\$", lambda scanner, token: "REGEND"),
         (r"\?", lambda scanner, token: "REGQUE"),
         (r"[\.~``;_a-zA-Z0-9\s=:\{\}\-\\]+", lambda scanner, token: "REFRE"),
         (r'.', lambda scanner, token: None)])

    def __init__(self):
        self.idCount = {"TABLE": 0, "COLUMN": 0}
        self.idMap = {"TABLE": {}, "COLUMN": {}}
        self.idMapInv = {}
        self.regex = False

    @staticmethod
    def sanitizeSql(sql):
        s = sql.strip().lower()
        if not s[-1] == ";":
            s += ';'
        s = re.sub(r'\(', r' ( ', s)
        s = re.sub(r'\)', r' ) ', s)
        words = ['index', 'table', 'day', 'year', 'user', 'text']
        for word in words:
            s = re.sub(r'([^\w])' + word + '$', r'\1' + word + '1', s)
            s = re.sub(r'([^\w])' + word + r'([^\w])', r'\1' + word + '1' + r'\2', s)
        s = s.replace('#', '')
        return s

    def tokenizeRegex(self, s):
        results = self.scanner.scan(s)[0]
        return results

    def parseStrings(self, tok):
        if isinstance(tok, sqlparse.sql.TokenList):
            for c in tok.tokens:
                self.parseStrings(c)
        elif tok.ttype == STRING:
            if self.regex:
                tok.value = ' '.join(self.tokenizeRegex(tok.value))
            else:
                tok.value = "CODSTR"

    def renameColumns(self, tok):
        if isinstance(tok, sqlparse.sql.TokenList):
            for c in tok.tokens:
                self.renameColumns(c)
        elif tok.ttype == COLUMN:
            normalized = tok.normalized
            if normalized not in self.idMap["COLUMN"]:
                colname = "col" + str(self.idCount["COLUMN"])
                self.idMap["COLUMN"][normalized] = colname
                self.idMapInv[colname] = normalized
                self.idCount["COLUMN"] += 1
            tok.value = self.idMap["COLUMN"][normalized]

    def renameTables(self, tok):
        if isinstance(tok, sqlparse.sql.TokenList):
            for c in tok.tokens:
                self.renameTables(c)
        elif tok.ttype == TABLE:
            normalized = tok.normalized
            if normalized not in self.idMap["TABLE"]:
                tabname = "tab" + str(self.idCount["TABLE"])
                self.idMap["TABLE"][normalized] = tabname
                self.idMapInv[tabname] = normalized
                self.idCount["TABLE"] += 1
            tok.value = self.idMap["TABLE"][normalized]

    def renameIdentifiers(self, tokens):
        for tok in tokens:
            self.renameColumns(tok)
            self.renameTables(tok)

    def parse(self, sql):
        sql = self.sanitizeSql(sql)
        parsed = sqlparse.parse(sql)[0]
        tokens = parsed.flatten()
        self.parseStrings(parsed)
        self.renameIdentifiers(tokens)
        return str(parsed)

    # Define token types
    INTERNAL = 'INTERNAL'
    SUBQUERY = 'SUBQUERY'
    KEYWORD = 'KEYWORD'
    INTEGER = 'INTEGER'
    HEX = 'HEX'
    FLOAT = 'FLOAT'
    STRING = 'STRING'
    WILDCARD = 'WILDCARD'
    COLUMN = 'COLUMN'
    TABLE = 'TABLE'
    FUNCTION = 'FUNCTION'

    class SqlangParser:
        def __init__(self, sql: str, regex: bool = False, rename: bool = True):
            """
            Parse SQL code and generate tokens.

            Args:
                sql: A string of SQL code.
                regex: A boolean indicating whether to use regular expressions in parsing.
                rename: A boolean indicating whether to rename identifiers in the parsed SQL.
            """
            self.sql = SqlangParser.sanitize_sql(sql)
            self.id_map = {"COLUMN": {}, "TABLE": {}}
            self.id_map_inv = {}
            self.id_count = {"COLUMN": 0, "TABLE": 0}
            self.regex = regex
            self.parse_tree_sentinel = False
            self.table_stack = []
            self.parse = sqlparse.parse(self.sql)
            self.parse = [self.parse[0]]
            self.remove_whitespaces(self.parse[0])
            self.identify_literals(self.parse[0])
            self.parse[0].ptype = SUBQUERY
            self.identify_sub_queries(self.parse[0])
            self.identify_functions(self.parse[0])
            self.identify_tables(self.parse[0])
            self.parse_strings(self.parse[0])
            if rename:
                self.rename_identifiers(self.parse[0])
            self.tokens = SqlangParser.get_tokens(self.parse)

        @staticmethod
        def sanitize_sql(sql: str) -> str:
            """
            Sanitize SQL code by removing comments and extra whitespace.

            Args:
                sql: A string of SQL code.

            Returns:
                A sanitized string of SQL code.
            """
            return sqlparse.format(sql, strip_comments=True, reindent=True)

        @staticmethod
        def get_tokens(parse) -> list:
            """
            Return a list of tokens extracted from a parse tree.

            Args:
                parse: A SQL parse tree.

            Returns:
                A list of tokens extracted from the parse tree.
            """
            flat_parse = []
            for expr in parse:
                for token in expr.flatten():
                    if token.ttype == sqlparse.tokens.STRING:
                        flat_parse.extend(str(token).split(' '))
                    else:
                        flat_parse.append(str(token))
            return flat_parse

        def remove_whitespaces(self, tok):
            """
            Remove whitespace tokens from a parse tree.

            Args:
                tok: A SQL parse tree.
            """
            if isinstance(tok, sqlparse.sql.TokenList):
                tmp_children = []
                for c in tok.tokens:
                    if not c.is_whitespace:
                        tmp_children.append(c)
                tok.tokens = tmp_children
                for c in tok.tokens:
                    self.remove_whitespaces(c)

        def identify_sub_queries(self, token_list):
            """
            Identify subqueries in a parse tree.

            Args:
                token_list: A SQL parse tree.
            """
            is_sub_query = False
            for tok in token_list.tokens:
                if isinstance(tok, sqlparse.sql.TokenList):
                    subquery = self.identify_sub_queries(tok)
                    if (subquery and isinstance(tok, sqlparse.sql.Parenthesis)):
                        tok.ttype = SUBQUERY
                elif str(tok) == "select":
                    is_sub_query = True
            return is_sub_query

        def identify_literals(self, token_list):
            """
            Identify literals in a parse tree.

            Args:
                token_list: A SQL parse tree.
            """
            blank_tokens = [sqlparse.tokens.Name, sqlparse.tokens.Name.Placeholder]
            blank_token_types = [sqlparse.sql.Identifier]
            for tok in token_list.tokens:
                if isinstance(tok, sqlparse.sql.TokenList):
                    tok.ptype = INTERNAL
                    self.identify_literals(tok)
                elif tok.ttype == sqlparse.tokens.Keyword or str(tok) == "select":
                    tok.ttype = KEYWORD
                elif tok.ttype in [sqlparse.tokens.Number.Integer, sqlparse.tokens.Literal.Number.Integer]:
                    tok.ttype = INTEGER
                elif tok.ttype in [sqlparse.tokens.Number.Hexadecimal, sqlparse.tokens.Literal.Number.Hexadecimal]:
                    tok.ttype = HEX
                elif tok.ttype in [sqlparse.tokens.Number.Float, sqlparse.tokens.Literal.Number.Float]:
                    tok.ttype = FLOAT
                elif tok.ttype in [sqlparse.tokens.String.Symbol, sqlparse.tokens.String.Single,
                                   sqlparse.tokens.Literal.String.Single, sqlparse.tokens.Literal.String.Symbol]:
                    tok.ttype = STRING
                elif tok.ttype == sqlparse.tokens.W

#############################################################################

#############################################################################
#缩略词处理

def revert_abbrev(line: str) -> str:
    """
    Revert abbreviations in a sentence.

    Args:
        line: A string representing a sentence.

    Returns:
        A string with abbreviations reverted.
    """
    pat_is = re.compile(r"(it|he|she|that|this|there|here)(\"s)", re.I)
    pat_s = re.compile(r"(?<=[a-zA-Z])\"s?")  # combine pat_s1 and pat_s2
    pat_not_would = re.compile(r"(?<=[a-zA-Z])n\"t|(?<=[a-zA-Z])\"d")  # combine pat_not and pat_would
    pat_will = re.compile(r"(?<=[a-zA-Z])\"ll")
    pat_am = re.compile(r"(?<=[I|i])\"m")
    pat_are = re.compile(r"(?<=[a-zA-Z])\"re")
    pat_ve = re.compile(r"(?<=[a-zA-Z])\"ve")

    line = pat_is.sub(r"\1 is", line)
    line = pat_s.sub("", line)
    line = pat_not_would.sub(lambda x: " not" if x.group(0) == "n\"t" else " would", line)
    line = pat_will.sub(" will", line)
    line = pat_am.sub(" am", line)
    line = pat_are.sub(" are", line)
    line = pat_ve.sub(" have", line)

    return line

def get_wordpos(tag: str) -> str:
    """
    Get the WordNet part of speech for a given Penn Treebank part of speech tag.

    Args:
        tag: A string representing a Penn Treebank part of speech tag.

    Returns:
        A string representing the WordNet part of speech, or None if the tag is not recognized.
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

# Define regular expressions as constants
DECIMAL_REGEX = re.compile(r"\d+(\.\d+)+")
STRING_REGEX = re.compile(r'\"[^\"]+\"')
HEX_REGEX = re.compile(r"0[xX][A-Fa-f0-9]+")
NUMBER_REGEX = re.compile(r"\s?\d+\s?")
OTHER_REGEX = re.compile(r"(?<![A-Z|a-z|_|])\d+[A-Za-z]+")

# Define string constants
TAG_INT = 'TAGINT'
TAG_STR = 'TAGSTR'
TAG_OER = 'TAGOER'

def process_nl_line(line: str) -> str:
    """
    Process a natural language sentence by removing unnecessary characters and converting to snake_case.

    Args:
        line: A string representing a natural language sentence.

    Returns:
        A string representing the processed sentence.
    """
    # Sentence preprocessing
    line = revert_abbrev(line)
    line = re.sub('\t+', '\t', line)
    line = re.sub('\n+', '\n', line)
    line = line.replace('\n', ' ')
    line = line.replace('\t', ' ')
    line = re.sub(' +', ' ', line)
    line = line.strip()

    # Convert camelCase to snake_case
    line = inflection.underscore(line)

    # Remove content inside parentheses
    space = re.compile(r"\([^\(|^\)]+\)")  # suffix matching
    line = re.sub(space, '', line)

    # Remove trailing period and whitespace
    line = line.strip()

    return line

def process_sent(line: str) -> list:
    """
    Process a sentence by tokenizing, replacing numbers and strings, and lemmatizing and stemming words.

    Args:
        line: A string representing a sentence.

    Returns:
        A list of tokens representing the processed sentence.
    """
    # Tokenize the sentence
    words = tokenize(line)

    # Replace numbers and strings with tags
    words = replace_numbers_and_strings(words)

    # Lemmatize and stem words
    words = lemmatize_and_stem(words)

    return words

def tokenize(line: str) -> list:
    """
    Tokenize a sentence by splitting on whitespace.

    Args:
        line: A string representing a sentence.

    Returns:
        A list of tokens representing the sentence.
    """
    words = re.findall(r"[\w]+|[^\s\w]", line)
    return words

def replace_numbers_and_strings(words: list) -> list:
    """
    Replace numbers and strings in a list of tokens with tags.

    Args:
        words: A list of tokens.

    Returns:
        A list of tokens with numbers and strings replaced by tags.
    """
    tags = []
    for word in words:
        if DECIMAL_REGEX.match(word):
            tags.append(TAG_INT)
        elif STRING_REGEX.match(word):
            tags.append(TAG_STR)
        elif HEX_REGEX.match(word):
            tags.append(TAG_INT)
        elif NUMBER_REGEX.match(word):
            tags.append(TAG_INT)
        elif OTHER_REGEX.match(word):
            tags.append(TAG_OER)
        else:
            tags.append(word)
    return tags

def lemmatize_and_stem(words: list) -> list:
    """
    Lemmatize and stem words in a list of tokens.

    Args:
        words: A list of tokens.

    Returns:
        A list of lemmatized and stemmed tokens.
        """
    word_list=[]
    for word in words:
        # Get the part of speech for the word
        word_pos = get_wordpos(tags_dict[word])
        if word_pos in ['a', 'v', 'n', 'r']:
            # Lemmatize the word
            word = wnler.lemmatize(word, pos=word_pos)
        # Stem the word
        word = wordnet.morphy(word) if wordnet.morphy(word) else word
        word_list.append(word)
    return word_list

def get_wordpos(tag: str) -> str:
    """
    Get the WordNet part of speech for a given Penn Treebank part of speech tag.

    Args:
        tag: A string representing a Penn Treebank part of speech tag.

    Returns:
        A string representing the WordNet part of speech, or None if the tag is not recognized.
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
#############################################################################

# Define regular expressions as constants
INVACHAR_REGEX_ALL = re.compile(r'[^(0-9|a-z|A-Z|\-|_|\'|\"|\-|\(|\)|\n)]+')
INVACHAR_REGEX_PART = re.compile(r'[^(0-9|a-z|A-Z|\-|#|/|_|,|\'|=|>|<|\"|\-|\\|\(|\)|\?|\.|\*|\+|\[|\]|\^|\{|\}|\n)]+')
NUMBER_REGEX = re.compile(r"\d+(\.\d+)+")

# Define string constants
NUMBER_TAG = 'number'
ERROR_TAG = '-1000'

def filter_all_invachar(line: str) -> str:
    """
    Filter out all non-alphanumeric characters in a string.

    Args:
        line: A string to be filtered.

    Returns:
        A string with all non-alphanumeric characters replaced by whitespace.
    """
    line = re.sub(INVACHAR_REGEX_ALL, ' ', line)
    line = re.sub('-+', '-', line)
    line = re.sub('_+', '_', line)
    line = line.replace('|', ' ').replace('¦', ' ')
    return line

def filter_part_invachar(line: str) -> str:
    """
    Filter out non-alphanumeric characters in a string, except for a specific set of characters.

    Args:
        line: A string to be filtered.

    Returns:
        A string with non-alphanumeric characters, except for a specific set of characters, replaced by whitespace.
    """
    line = re.sub(INVACHAR_REGEX_PART, ' ', line)
    line = re.sub('-+', '-', line)
    line = re.sub('_+', '_', line)
    line = line.replace('|', ' ').replace('¦', ' ')
    return line

def parse_sqlang_code(line: str) -> list or str:
    """
    Parse SQL code and return a list of tokens representing the parsed code.

    Args:
        line: A string representing SQL code.

    Returns:
        A list of tokens representing the parsed SQL code, or the string '-1000' if parsing fails.
    """
    line = filter_part_invachar(line)
    line = re.sub('\.+', '.', line)
    line = re.sub('\t+', '\t', line)
    line = re.sub('\n+', '\n', line)
    line = re.sub(' +', ' ', line)

    line = re.sub('>>+', '', line)
    line = re.sub(NUMBER_REGEX, NUMBER_TAG, line)

    line = line.strip('\n').strip()
    line = re.findall(r"[\w]+|[^\s\w]", line)
    line = ' '.join(line)

    try:
        query = SqlangParser(line, regex=True)
        typed_code = query.parseSql()
        typed_code = typed_code[:-1]
        # Convert camelCase to snake_case
        typed_code = inflection.underscore(' '.join(typed_code)).split(' ')

        cut_tokens = [re.sub("\s+", " ", x.strip()) for x in typed_code]
        # Convert all tokens to lowercase
        token_list = [x.lower() for x in cut_tokens if x.strip() != '']
        # Return the list of tokens
        return token_list
    except Exception as e:
        # Handle exceptions by returning an error tag
        return ERROR_TAG
########################主函数：代码的tokens#################################


#######################主函数：句子的tokens##################################

# Define string constants
EMPTY_STRING = ''
SPACE_STRING = ' '

def sqlang_query_parse(line: str) -> list:
    """
    Parse a string representing a SQL query and return a list of tokens representing the parsed query.

    Args:
        line: A string representing a SQL query.

    Returns:
        A list of tokens representing the parsed SQL query.
    """
    line = filter_all_invachar(line)
    line = process_nl_line(line)
    word_list = process_sent_word(line)
    # Remove parentheses from the token list
    for i in range(0, len(word_list)):
        if re.findall('[\(\)]', word_list[i]):
            word_list[i] = EMPTY_STRING
    # Remove empty or space-only tokens from the token list
    word_list = [x.strip() for x in word_list if x.strip() != EMPTY_STRING]
    return word_list

def sqlang_context_parse(line: str) -> list:
    """
    Parse a string representing SQL context and return a list of tokens representing the parsed context.

    Args:
        line: A string representing SQL context.

    Returns:
        A list of tokens representing the parsed SQL context.
    """
    line = filter_part_invachar(line)
    line = process_nl_line(line)
    word_list = process_sent_word(line)
    # Remove empty or space-only tokens from the token list
    word_list = [x.strip() for x in word_list if x.strip() != EMPTY_STRING]
    return word_list
#######################主函数：句子的tokens##################################

if __name__ == '__main__':
    print(sqlang_code_parse('""geometry": {"type": "Polygon" , 111.676,"coordinates": [[[6.69245274714546, 51.1326962505233], [6.69242714158622, 51.1326908883821], [6.69242919794447, 51.1326955158344], [6.69244041615532, 51.1326998744549], [6.69244125953742, 51.1327001609189], [6.69245274714546, 51.1326962505233]]]} How to 123 create a (SQL  Server function) to "join" multiple rows from a subquery into a single delimited field?'))
    print(sqlang_query_parse("change row_height and column_width in libreoffice calc use python tagint"))
    print(sqlang_query_parse('MySQL Administrator Backups: "Compatibility Mode", What Exactly is this doing?'))
    print(sqlang_code_parse('>UPDATE Table1 \n SET Table1.col1 = Table2.col1 \n Table1.col2 = Table2.col2 FROM \n Table2 WHERE \n Table1.id =  Table2.id'))
    print(sqlang_code_parse("SELECT\n@supplyFee:= 0\n@demandFee := 0\n@charedFee := 0\n"))
    print(sqlang_code_parse('@prev_sn := SerialNumber,\n@prev_toner := Remain_Toner_Black\n'))
    print(sqlang_code_parse(' ;WITH QtyCTE AS (\n  SELECT  [Category] = c.category_name\n          , [RootID] = c.category_id\n          , [ChildID] = c.category_id\n  FROM    Categories c\n  UNION ALL \n  SELECT  cte.Category\n          , cte.RootID\n          , c.category_id\n  FROM    QtyCTE cte\n          INNER JOIN Categories c ON c.father_id = cte.ChildID\n)\nSELECT  cte.RootID\n        , cte.Category\n        , COUNT(s.sales_id)\nFROM    QtyCTE cte\n        INNER JOIN Sales s ON s.category_id = cte.ChildID\nGROUP BY cte.RootID, cte.Category\nORDER BY cte.RootID\n'))
    print(sqlang_code_parse("DECLARE @Table TABLE (ID INT, Code NVARCHAR(50), RequiredID INT);\n\nINSERT INTO @Table (ID, Code, RequiredID)   VALUES\n    (1, 'Physics', NULL),\n    (2, 'Advanced Physics', 1),\n    (3, 'Nuke', 2),\n    (4, 'Health', NULL);    \n\nDECLARE @DefaultSeed TABLE (ID INT, Code NVARCHAR(50), RequiredID INT);\n\nWITH hierarchy \nAS (\n    --anchor\n    SELECT  t.ID , t.Code , t.RequiredID\n    FROM @Table AS t\n    WHERE t.RequiredID IS NULL\n\n    UNION ALL   \n\n    --recursive\n    SELECT  t.ID \n          , t.Code \n          , h.ID        \n    FROM hierarchy AS h\n        JOIN @Table AS t \n            ON t.RequiredID = h.ID\n    )\n\nINSERT INTO @DefaultSeed (ID, Code, RequiredID)\nSELECT  ID \n        , Code \n        , RequiredID\nFROM hierarchy\nOPTION (MAXRECURSION 10)\n\n\nDECLARE @NewSeed TABLE (ID INT IDENTITY(10, 1), Code NVARCHAR(50), RequiredID INT)\n\nDeclare @MapIds Table (aOldID int,aNewID int)\n\n;MERGE INTO @NewSeed AS TargetTable\nUsing @DefaultSeed as Source on 1=0\nWHEN NOT MATCHED then\n Insert (Code,RequiredID)\n Values\n (Source.Code,Source.RequiredID)\nOUTPUT Source.ID ,inserted.ID into @MapIds;\n\n\nUpdate @NewSeed Set RequiredID=aNewID\nfrom @MapIds\nWhere RequiredID=aOldID\n\n\n/*\n--@NewSeed should read like the following...\n[ID]  [Code]           [RequiredID]\n10....Physics..........NULL\n11....Health...........NULL\n12....AdvancedPhysics..10\n13....Nuke.............12\n*/\n\nSELECT *\nFROM @NewSeed\n"))



