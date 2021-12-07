# question generator
from .utils import *
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import random
import os
from deep_translator import GoogleTranslator

class QuestionGenerator:
    """
    This is the class for question generation.
    """
    def __init__(self, lang=None):
        def set_seed(seed):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        set_seed(42)

        self.model = T5ForConditionalGeneration.from_pretrained(os.path.join(os.getcwd(), 'MISP_SQL/t5_paraphrase'))
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.lang = lang

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print ("device ",self.device)
        self.model = self.model.to(self.device)

        # the seed lexicon
        self.agg_regular = {"avg": "the average value of",
                            "count": "the number of items in",
                            "sum": "the sum of values of",
                            "min": "the minimum value among items of",
                            "max": "the maximum value among items of"}
        self.agg_distinct = {"avg": "the average value of distinct items in",
                             "count": "the number of distinct items in",
                             "sum": "the sum of distinct values of",
                             "min": "the minimum value among distinct items of",
                             "max": "the maximum value among distinct items of"}
        self.agg_asterisk = {"avg": "the average value of items", # warning: this should not be triggered
                             "count": "the number of items",
                             "sum": "the sum of values", # warning: this should not be triggered
                             "min": "the minimum value among items", # warning: this should not be triggered
                             "max": "the maximum value among items"} # warning: this should not be triggered
        self.agg_asterisk_distinct = {"avg": "the average value of distinct items",  # warning: this should not be triggered
                                      "count": "the number of distinct items",
                                      "sum": "the sum of distinct values",  # warning: this should not be triggered
                                      "min": "the minimum value among distinct items",  # warning: this should not be triggered
                                      "max": "the maximum value among distinct items"}  # warning: this should not be triggered
        self.where_op = {"like": "follow", "not in": "be NOT IN", ">": "be greater than",
                         "<": "be less than", "=": "equal to", ">=": "be greater than or equal to",
                         "<=": "be less than or equal to", "!=": "be NOT equal to",
                         "in": "be IN", "between": "be between"}
        self.desc_asc_limit = {("desc", False): "in descending order", ("asc", False): "in ascending order",
                               ("desc", True): "in descending order and limited to top N",
                               ("asc", True): "in ascending order and limited to top N"}

    def set_lang(self, lang):
        self.lang = lang


    def paraphrase_question(self,sentence):
        #sentence = "Which course should I take to get started in data science?"
        text =  "paraphrase: " + sentence + " </s>"
        max_len = 256
        encoding = self.tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)


        # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
        beam_outputs = self.model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            do_sample=True,
            max_length=256,
            top_k=120,
            top_p=0.98,
            early_stopping=True,
            num_return_sequences=10)


        #print ("\nOriginal Question ::")
        #print (sentence)
        #print ("\n")
        #print ("Paraphrased Questions :: ")
        final_outputs =[]
        for beam_output in beam_outputs:
            sent = self.tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
            if sent.lower() != sentence.lower() and sent not in final_outputs:
                final_outputs.append(sent)

        sentence_op = GoogleTranslator(source='auto', target=self.lang).translate(final_outputs[0])
        print(sentence_op)
        return sentence_op
        #return final_outputs[random.randint(0, len(final_outputs)-1)]

    def agg_col_tab_description(self, col_name, tab_name, agg=None, bool_having=False, bool_distinct=False):
        """
        Creates a description for the aggregation preformed on a attribute for a table.
        :param col_name: The attribute of the items to consider for agg. type: str
        :param tab_name: The table to take items from. type: str
        :param agg: The type of aggregation to preform. type: ("avg", "count", "sum", "min", "max") -> str
        :param bool_having: If the table is made up of groups. type: bool
        :param bool_distinct: If the items should be distinct or not. type: bool
        """
        if agg is not None:
            agg = agg.lower()

        if bool_distinct:
            assert agg is not None

        if col_name == "*":
            agg_descrip = "all items"
            if agg is not None and bool_distinct:
                agg_descrip = self.agg_asterisk_distinct[agg]
            elif agg is not None:
                agg_descrip = self.agg_asterisk[agg]

            tab_descrip = ""
            if bool_having:
                tab_descrip += " in each group"
            #Removing colors
            if tab_name is not None:
                tab_descrip += " in the table " + tab_name

            return agg_descrip + tab_descrip
        else:
            agg_descrip = ""
            if agg is not None and bool_distinct:
                agg_descrip = self.agg_distinct[agg] + " "
            elif agg is not None:
                agg_descrip = self.agg_regular[agg] + " "

            col_descrip = "the data " + col_name

            tab_descrip = " in the table"
            if tab_name is not None:
                tab_descrip += " " + tab_name
            return agg_descrip + col_descrip + tab_descrip

    def group_by_agg_col_tab_description(self, col_name, tab_name):
        """
        Creates a description for a group in a table of the aggregation of an attribute. 
        :param col_name: The attribute of the items to consider for agg. type: str
        :param tab_name: The table to take items from. type: str
        """
        if tab_name is None:
            return "finding based on the table data " + col_name
        else:
            return "finding in the table " + tab_name +\
                   " based on the table data " + col_name

    def select_col_question(self, col_name, tab_name):
        return "Please confirm if you need information on %s?" % self.agg_col_tab_description(col_name, tab_name)

    def select_agg_question(self, col_name, tab_name, src_agg, bool_distinct=False):
        """
        Asks for clarification about the aggregation. 
        :param col_name: The attribute of the items to consider for agg. type: str
        :param tab_name: The table to take items from. type: str
        :param src_agg: The original aggregation performed. type: ("avg", "count", "sum", "min", "max") -> str
        :param bool_distinct: If the table members should be distinct or not. type: bool
        """
        if src_agg == "none_agg":
            return "Shall I give detail about %s?" % self.agg_col_tab_description(col_name, tab_name)
        else:
            src_agg = src_agg.lower()
            return "Shall I give detail about %s?" % self.agg_col_tab_description(
                col_name, tab_name, agg=src_agg, bool_distinct=bool_distinct)

    def where_col_question(self, col_name, tab_name):
        return "I have a confusion!, should I consider conditions associated with %s?" %\
               self.agg_col_tab_description(col_name, tab_name)

    def andor_question(self, and_or, selected_cols_info): # deprecated
        """
        Asks if the attribute data is synced or not.
        :param and_or: String representing either and or or. type: str
        :param selected_cols_info: Information related to the selected attributes. type: str
        """
        if and_or == "and":
            return "I found some conditions on %s, do they apply at same time?" % selected_cols_info
        elif and_or == "or":
            return "I found some conditions on %s, do they apply alternatively?" % selected_cols_info
        else:
            raise ValueError("Invalid and_or=%s!" % and_or)

    def where_op_question(self, agg_col_tab_name, op_name):
        """
        Asks if the the resulting agg_col_tab_name should have a condition applied to it.
        :param agg_col_tab_name: The value of a aggregation on an attribute of a table. type: str
        :param op_name: The operation that represents the condition. type: (like, not in, >, <, =, >=, <=, !=, in, between) -> str
        """
        value_descrip = "patterns" if op_name == "like" else "values"
        return "I am enforcing the condition that in the results, %s must %s some specific %s. " % (
            agg_col_tab_name, self.where_op[op_name], value_descrip) + "Is the condition correct?"

    def root_terminal_question(self, col_name, tab_name, op_name, root_or_terminal,
                               bool_having=False, agg=None, group_by_col_info=None,
                               bool_distinct=False):
        """
        Confirms with the user that a specific condition over an atribute of a table
        or group of tables is correct.
        :param col_name: The attribute the condition will apply to. type: str
        :param tab_name: The name of the table that the condition will apply to. type: str
        :param op_name: The name of the operation of the condition. type: (like, not in, >, <, =, >=, <=, !=, in, between) -> str
        :param root_or_terminal: Values VS. Literal values respectivly. type: (root, terminal) -> str 
        :param bool_having: If the table is made up of groups. type: bool 
        :param agg: The type of aggregation to preform. type: ("avg", "count", "sum", "min", "max") -> str
        :param group_by_col_info: If the values should be grouped by attribute information. type: bool
        :param bool_distinct: If the items should be distinct or not. type: bool
        """
        root_term_description = self.root_terminal_description(col_name, tab_name, op_name, root_or_terminal,
                                                               bool_having=bool_having, agg=agg,
                                                               bool_distinct=bool_distinct)

        if bool_having:
            question = "I will group %s. " \
                       "Can I am applying the filter conditions that in the results, %s?" % (
                       group_by_col_info, root_term_description)
        else:
            question = "I am applying the filter condition that in the results, %s. Is this condition correct?" % (
                root_term_description)

        return question

    def root_terminal_description(self, col_name, tab_name, op_name, root_or_terminal,
                                  bool_having=False, agg=None, bool_distinct=False):
        agg_col_tab_name = self.agg_col_tab_description(col_name, tab_name, agg=agg, bool_having=bool_having,
                                                        bool_distinct=bool_distinct)
        """
        Generates a description of what conditions will be applied to attribute and table.
        :param col_name: The attribute the condition will apply to. type: str
        :param tab_name: The name of the table that the condition will apply to. type: str
        :param op_name: The name of the operation of the condition. type: (like, not in, >, <, =, >=, <=, !=, in, between) -> str
        :param root_or_terminal: Values VS. Literal values respectivly. type: (root, terminal) -> str 
        :param bool_having: If the table is made up of groups. type: bool 
        :param agg: The type of aggregation to preform. type: ("avg", "count", "sum", "min", "max") -> str
        :param bool_distinct: If the items should be distinct or not. type: bool
        """
        if root_or_terminal == "terminal":
            if op_name in {"in", "not in"}:
                value_descrip = "a list of given values (e.g., number 5, string \"France\")"
            elif op_name == "between":
                value_descrip = "two given values (e.g., number 5, string \"France\")"
            else:
                value_descrip = "a given value (e.g., number 5, string \"France\")"
        else:
            assert root_or_terminal == "root"
            if op_name in {"in", "not in"}:
                value_descrip = "a set of values to be calculated"
            else:
                value_descrip = "a value to be calculated"

        description = "%s must %s %s" % (agg_col_tab_name, self.where_op[op_name], value_descrip)

        return description

    def where_val_question(self, col_name, tab_name, op_name, val_str):
        return "I am applying conditions in the results, %s must %s %s. " % (
            self.agg_col_tab_description(col_name, tab_name), self.where_op[op_name], val_str) + \
               "Is the condition correct?"

    def group_col_question(self, col_name, tab_name):
        """
        Confirms with the user if attributes need to be grouped.
        :param col_name: The attribute that may need grouping. type: str
        :param tab_name: The table the attribute comes from. type: str
        """
        assert tab_name is not None
        return "Do I need to group %s?" % self.group_by_agg_col_tab_description(col_name, tab_name)

    def group_none_having_question(self, group_by_cols_info): # deprecated
        return "I shall group %s, but " % group_by_cols_info + "without" + \
               " considering any other conditions. Is this correct?"

    def have_col_question(self, group_by_cols_info, col_name, tab_name):
        question = "I shall first group %s. " \
                   "Can I consider any specific conditions about %s?" % (
            group_by_cols_info, self.agg_col_tab_description(col_name, tab_name, bool_having=True))

        return question

    def have_agg_question(self, group_by_cols_info, col_name, tab_name, src_agg, bool_distinct=False):
        """
        Asks user if they want to add conditions to a different set of aggregations. Adds that the 
        table is grouped in some way. AKA Having = True.
        :param group_by_cols_info: The info of how we are grouping previous attributes. type: str
        :param col_name: The attributes that agg will effect. type: src
        :param tab_name: The table that the attributes are from. type: src
        :param src_agg: The aggregation that will be applied to the attributes. type: ("avg", "count", "sum", "min", "max") -> str
        :param bool_distinct: TODO *** NOT NEEDED, UNUSED ***
        """
        src_agg = src_agg.lower()
        if src_agg == "none_agg":
            question = "I shall first group %s. " \
                       "Can I consider any specific conditions about the value of %s?" % (
                           group_by_cols_info, self.agg_col_tab_description(col_name, tab_name, bool_having=True))

        else:
            agg = src_agg

            question = "I shall first group %s. " \
                       "Can I consider any specific conditions about %s?" % (
                           group_by_cols_info, self.agg_col_tab_description(col_name, tab_name, agg=agg, bool_having=True))

        return question

    def have_op_question(self, group_by_cols_info, col_name, tab_name, op_name, agg=None, bool_distinct=False):
        """
        Asks the user if some condition needs to be enforced onto the results of agg. Adds that the 
        table is grouped in some way. AKA Having = True.
        :param group_by_cols_info: The info of how we are grouping previous attributes. type: str
        :param col_name: The attributes being affected by agg. type: src
        :param tab_name: The table the attributes are from. type: src
        :param op_name: The operation that will be used on the results of agg. type: (like, not in, >, <, =, >=, <=, !=, in, between) -> str
        :param agg: The aggregation used on the attributes. type: ("avg", "count", "sum", "min", "max") -> str
        :param bool_distinct: If the groups/attributes should be distinct. type: bool
        """
        value_descrip = "patterns" if op_name == "like" else "values"
        question = "I shall first group %s. " \
                   "Can I apply condition in the results, " \
                   "%s must % some specific %s?" % (
            group_by_cols_info, self.agg_col_tab_description(col_name, tab_name, agg=agg,
                                                             bool_having=True, bool_distinct=bool_distinct),
            self.where_op[op_name], value_descrip)

        return question

    def order_col_question(self, col_name, tab_name):
        return "Shall I order " \
               "the results based on %s?" % self.agg_col_tab_description(col_name, tab_name)

    def order_agg_question(self, col_name, tab_name, src_agg, bool_distinct=False):
        """
        Asks the user if the reuslts of a aggregation should be sorted in a particular order.
        :param col_name: The attributes the aggregation is performed on. type: str
        :param tab_name: The table the attributes come from. type: str
        :param src_agg: The aggregation to apply to the attributes. type: (like, not in, >, <, =, >=, <=, !=, in, between) -> str
        :param bool_distinct: I the results should be distinct or not: type: bool
        """
        src_agg = src_agg.lower()
        if src_agg == "none_agg":
            return "Shall I order " \
                   "the results based on the value of %s?" % self.agg_col_tab_description(col_name, tab_name)
        else:
            agg = src_agg
            return "Shall I order " \
                   "the results based on %s?" % self.agg_col_tab_description(col_name, tab_name, agg=agg,
                                                                             bool_distinct=bool_distinct)

    def order_desc_asc_limit_question(self, col_name, tab_name, desc_asc_limit, agg=None):
        return "I think I would sort the results based on %s. \n" \
               "\tIf this assumption is correct, do the results need to be %s? \n" \
               "\tIf you think this assumption is incorrect, please select the 'No' option." % (
            self.agg_col_tab_description(col_name, tab_name, agg=agg), self.desc_asc_limit[desc_asc_limit])

    def order_desc_asc_question(self, col_name, tab_name, desc_asc, agg=None, bool_distinct=False):
        return "I think I would sort the results based on %s. \n" \
               "\tIf this assumption is correct, do the results need to be %s? \n" \
               "\tIf you think this assumption is incorrect, please select the 'No' option." % (
                   self.agg_col_tab_description(col_name, tab_name, agg=agg, bool_distinct=bool_distinct),
                   self.desc_asc_limit[(desc_asc, False)])

    def order_limit_question(self, col_name, tab_name, agg=None, bool_distinct=False):
        return "I think I would sort the results based on %s (in ascending or descending order). \n" \
               "\tIf this assumption is true, do the results need to be limited to top N? \n" \
               "\tIf you think this assumption is incorrect, please select the 'No' option." % (
            self.agg_col_tab_description(col_name, tab_name, agg=agg, bool_distinct=bool_distinct))

    def iuen_question(self, iuen):
        """
        Asks user if it needs to merge/compare different groups.
        :param iuen: How it should compare the groups. type: (except, union, intersect, ?) -> str
        """
        if iuen == "except":
            return "Do I need to return information satisfying some cases BUT NOT others?\n" \
                   "e.g., Find all airlines that have flights from airport 'CVO' BUT NOT from 'APG'."
        elif iuen == "union":
            return "Do I need to return information satisfying either some cases OR others?\n" \
                   "e.g., What are the id and names of the countries which have more than 3 car makers OR " \
                   "produce the 'fiat' model?"
        elif iuen == "intersect":
            return "Do I need to return information satisfying BOTH some cases AND the others AT THE " \
                   "SAME TIME?\ne.g., Which district has BOTH stores with less than 3000 products AND " \
                   "stores with more than 10000 products?"
        else:
            return "Do I need to return information that meets one of the three situations: \n" \
                   "(1) satisfying some cases BUT NOT others, e.g., Find all airlines that have flights " \
                   "from airport 'CVO' BUT NOT from 'APG'.\n" \
                   "(2) satisfying either some cases OR others, e.g., What are the id and " \
                   "names of the countries which have more than 3 car makers OR produce the 'fiat' model?\n" \
                   "(3) satisfying BOTH some cases AND the others AT THE SAME TIME, e.g., Which district has BOTH " \
                   "stores with less than 3000 products AND stores with more than 10000 products?\n" \
                   "(Note: your situation is very likely to fall into NONE of the above - suggest to answer 'no')"

    def where_having_nested_question(self, col_name, tab_name, op_name, right_question, agg=None, bool_having=False,
                                     bool_distinct=False):
        """
        Calculates the updated values using root_terminal and displays the corected description.
        :param col_name: The attributes affected by the aggregation. type: str
        :param tab_name: The table the attributes are from. type: str
        :param op_name: The operation that represents the condition. type: (like, not in, >, <, =, >=, <=, !=, in, between) -> str
        :param right_question: TODO: I don't understand this.
        :param agg: The aggregation performed on col_name. type: (like, not in, >, <, =, >=, <=, !=, in, between) -> str
        :param bool_having: If the table is made up of groups. type: bool
        """
        revised_right_question = right_question[:-1] + " for this calculation?"
        return "Assume the system will enforce the condition that in the results, %s, " \
               "answer the following question to help the system to calculate " % self.root_terminal_description(
               col_name, tab_name, op_name, "root", agg=agg, bool_having=bool_having) +\
               "the value(s)" + ": \n%s" % revised_right_question

    def question_generation(self, semantic_unit, tag_seq, pointer):
        """
        Generating NL questions.
        :param semantic_unit: the questioned semantic unit.
        :param tag_seq: the tag sequence produced by the parser.
        :param pointer: the pointer to tag_item in the tag_seq.
        :return: an NL question and cheat_sheet = {'yes'/'no': (bool_correct, bool_binary_choice_unit)}, where
                 bool_correct is True when the the user response ('yes' or 'no') means the decision is correct, and
                 bool_binary_choice_unit is True when there are only two choices for the decision (e.g., AND/OR); in
                 this case, the agent will switch to the alternative choice when the current one is regared wrong.
                 In general, cheat_sheet is used to simulate user feedback. For example, {'yes': (True, None), 'no': (False, 0)}
                 indicates that, if the user answers 'yes', she affirms the decision; if she answers 'no', she negates it.
        """
        assert tag_seq[pointer] == semantic_unit

        semantic_tag = semantic_unit[0]
        if semantic_tag == SELECT_COL:
            tab_col_item = semantic_unit[1]
            question = self.select_col_question(tab_col_item[1], tab_col_item[0])
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == SELECT_AGG:
            col, agg = semantic_unit[1:3]
            question = self.select_agg_question(col[1], col[0], agg[0])
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == SELECT_AGG_v2:
            col, agg, bool_distinct = semantic_unit[1:4]
            question = self.select_agg_question(col[1], col[0], agg[0], bool_distinct=bool_distinct)
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == WHERE_COL:
            tab_col_item = semantic_unit[1]
            question = self.where_col_question(tab_col_item[1], tab_col_item[0])
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == ANDOR:
            and_or, cols = semantic_unit[1:3]
            cols_info = [self.agg_col_tab_description(col[1], col[0]) for col in cols]
            question = self.andor_question(and_or, ", ".join(cols_info))
            cheat_sheet = {'yes': (True, None), 'no': (False, 1)}

        elif semantic_tag == WHERE_OP:
            ((col,), op) = semantic_unit[1:3]
            question = self.where_op_question(self.agg_col_tab_description(col[1], col[0]), op[0])
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == WHERE_VAL:
            ((col,), op, val_item) = semantic_unit[1:4]
            question = self.where_val_question(col[1], col[0], op[0], val_item[-1])
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == WHERE_ROOT_TERM:
            ((col,), op, root_term) = semantic_unit[1:4]
            question = self.root_terminal_question(col[1], col[0], op[0], root_term)
            cheat_sheet = {'yes': (True, None), 'no': (False, 1)}

        elif semantic_tag == GROUP_COL:
            tab_col_item = semantic_unit[1]
            question = self.group_col_question(tab_col_item[1], tab_col_item[0])
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}  # no->drop

        elif semantic_tag == GROUP_NHAV:
            groupBy_cols = []
            # idx = pointer - 2
            idx = helper_find_closest_bw(tag_seq, pointer - 1, tgt_id=GROUP_COL)
            while idx > 0:
                if tag_seq[idx][0] == GROUP_COL:
                    groupBy_cols.append(self.group_by_agg_col_tab_description(tag_seq[idx][1][1], tag_seq[idx][1][0]))
                else:
                    break
                idx -= 1
            question = self.group_none_having_question(", ".join(groupBy_cols))
            cheat_sheet = {'yes': (True, None), 'no': (False, 1)}

        elif semantic_tag == HAV_COL:
            tab_col_item = semantic_unit[1]

            closest_group_col_idx = helper_find_closest_bw(tag_seq, pointer - 1, tgt_id=GROUP_COL)
            groupBy_cols = []
            idx = closest_group_col_idx
            while idx > 0:
                if tag_seq[idx][0] == GROUP_COL:
                    groupBy_cols.append(self.group_by_agg_col_tab_description(tag_seq[idx][1][1], tag_seq[idx][1][0]))
                else:
                    break
                idx -= 1

            if len(groupBy_cols) > 1:
                group_by_col_info = ", ".join(groupBy_cols[:-1]) + " and " + groupBy_cols[-1]
            else:
                group_by_col_info = groupBy_cols[0]

            question = self.have_col_question(group_by_col_info, tab_col_item[1], tab_col_item[0])
            cheat_sheet = {'yes': (True, None), 'no':(False, 0)}

        elif semantic_tag == HAV_AGG:
            col, agg = semantic_unit[1:3]

            closest_group_col_idx = helper_find_closest_bw(tag_seq, pointer - 1, tgt_id=GROUP_COL)
            groupBy_cols = []
            idx = closest_group_col_idx
            while idx > 0:
                if tag_seq[idx][0] == GROUP_COL:
                    groupBy_cols.append(self.group_by_agg_col_tab_description(tag_seq[idx][1][1], tag_seq[idx][1][0]))
                else:
                    break
                idx -= 1

            if len(groupBy_cols) > 1:
                group_by_col_info = ", ".join(groupBy_cols[:-1]) + " and " + groupBy_cols[-1]
            else:
                group_by_col_info = groupBy_cols[0]

            question = self.have_agg_question(group_by_col_info, col[1], col[0], agg[0])

            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == HAV_AGG_v2:
            col, agg, bool_distinct = semantic_unit[1:4]
            closest_group_col_idx = helper_find_closest_bw(tag_seq, pointer - 1, tgt_id=GROUP_COL)
            groupBy_cols = []
            idx = closest_group_col_idx
            while idx > 0:
                if tag_seq[idx][0] == GROUP_COL:
                    groupBy_cols.append(self.group_by_agg_col_tab_description(tag_seq[idx][1][1], tag_seq[idx][1][0]))
                else:
                    break
                idx -= 1

            if len(groupBy_cols) > 1:
                group_by_col_info = ", ".join(groupBy_cols[:-1]) + " and " + groupBy_cols[-1]
            else:
                group_by_col_info = groupBy_cols[0]

            question = self.have_agg_question(group_by_col_info, col[1], col[0], agg[0],
                                              bool_distinct=bool_distinct)

            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == HAV_OP:
            (col, agg), op = semantic_unit[1:3]

            closest_group_col_idx = helper_find_closest_bw(tag_seq, pointer - 1, tgt_id=GROUP_COL)
            groupBy_cols = []
            idx = closest_group_col_idx
            while idx > 0:
                if tag_seq[idx][0] == GROUP_COL:
                    groupBy_cols.append(self.group_by_agg_col_tab_description(tag_seq[idx][1][1], tag_seq[idx][1][0]))
                else:
                    break
                idx -= 1

            if len(groupBy_cols) > 1:
                group_by_col_info = ", ".join(groupBy_cols[:-1]) + " and " + groupBy_cols[-1]
            else:
                group_by_col_info = groupBy_cols[0]

            question = self.have_op_question(group_by_col_info, col[1], col[0], op[0],
                                             agg=None if agg[0] == "none_agg" else agg[0])
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == HAV_OP_v2:
            (col, agg, bool_distinct), op = semantic_unit[1:3]

            closest_group_col_idx = helper_find_closest_bw(tag_seq, pointer - 1, tgt_id=GROUP_COL)
            groupBy_cols = []
            idx = closest_group_col_idx
            while idx > 0:
                if tag_seq[idx][0] == GROUP_COL:
                    groupBy_cols.append(self.group_by_agg_col_tab_description(tag_seq[idx][1][1], tag_seq[idx][1][0]))
                else:
                    break
                idx -= 1

            if len(groupBy_cols) > 1:
                group_by_col_info = ", ".join(groupBy_cols[:-1]) + " and " + groupBy_cols[-1]
            else:
                group_by_col_info = groupBy_cols[0]

            question = self.have_op_question(group_by_col_info, col[1], col[0], op[0],
                                             agg=None if agg[0] == "none_agg" else agg[0],
                                             bool_distinct=bool_distinct)
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == HAV_ROOT_TERM:
            (col, agg), op, root_term = semantic_unit[1:4]

            closest_group_col_idx = helper_find_closest_bw(tag_seq, pointer - 1, tgt_id=GROUP_COL)
            groupBy_cols = []
            idx = closest_group_col_idx
            while idx > 0:
                if tag_seq[idx][0] == GROUP_COL:
                    groupBy_cols.append(self.group_by_agg_col_tab_description(tag_seq[idx][1][1], tag_seq[idx][1][0]))
                else:
                    break
                idx -= 1

            if len(groupBy_cols) > 1:
                group_by_col_info = ", ".join(groupBy_cols[:-1]) + " and " + groupBy_cols[-1]
            else:
                group_by_col_info = groupBy_cols[0]

            question = self.root_terminal_question(col[1], col[0], op[0], root_term, bool_having=True,
                                                   agg=None if agg[0] == "none_agg" else agg[0],
                                                   group_by_col_info=group_by_col_info)
            cheat_sheet = {'yes': (True, None), 'no': (False, 1)}

        elif semantic_tag == HAV_ROOT_TERM_v2:
            (col, agg, bool_distinct), op, root_term = semantic_unit[1:4]

            closest_group_col_idx = helper_find_closest_bw(tag_seq, pointer - 1, tgt_id=GROUP_COL)
            groupBy_cols = []
            idx = closest_group_col_idx
            while idx > 0:
                if tag_seq[idx][0] == GROUP_COL:
                    groupBy_cols.append(self.group_by_agg_col_tab_description(tag_seq[idx][1][1], tag_seq[idx][1][0]))
                else:
                    break
                idx -= 1

            if len(groupBy_cols) > 1:
                group_by_col_info = ", ".join(groupBy_cols[:-1]) + " and " + groupBy_cols[-1]
            else:
                group_by_col_info = groupBy_cols[0]

            question = self.root_terminal_question(col[1], col[0], op[0], root_term, bool_having=True,
                                                   agg=None if agg[0] == "none_agg" else agg[0],
                                                   group_by_col_info=group_by_col_info,
                                                   bool_distinct=bool_distinct)
            cheat_sheet = {'yes': (True, None), 'no': (False, 1)}

        elif semantic_tag == ORDER_COL:
            tab_col_item = semantic_unit[1]
            question = self.order_col_question(tab_col_item[1], tab_col_item[0])
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == ORDER_AGG:
            col, agg = semantic_unit[1:3]
            question = self.order_agg_question(col[1], col[0], agg[0])
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == ORDER_AGG_v2:
            col, agg, bool_distinct = semantic_unit[1:4]
            question = self.order_agg_question(col[1], col[0], agg[0], bool_distinct=bool_distinct)
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == ORDER_DESC_ASC_LIMIT:
            (col, agg), desc_asc_limit = semantic_unit[1:3]
            question = self.order_desc_asc_limit_question(col[1], col[0], desc_asc_limit,
                                                          agg=None if agg[0] == "none_agg" else agg[0])
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == ORDER_DESC_ASC:
            (col, agg, bool_distinct), desc_asc = semantic_unit[1:3]
            question = self.order_desc_asc_question(col[1], col[0], desc_asc,
                                                    agg=None if agg[0] == "none_agg" else agg[0],
                                                    bool_distinct=bool_distinct)
            cheat_sheet = {'yes': (True, None), 'no': (False, 1)}

        elif semantic_tag == ORDER_LIMIT:
            (col, agg, bool_distinct) = semantic_unit[1]
            question = self.order_limit_question(col[1], col[0],
                                                 agg=None if agg[0] == "none_agg" else agg[0],
                                                 bool_distinct=bool_distinct)
            cheat_sheet = {'yes': (True, None), 'no': (False, 1)}

        elif semantic_tag == IUEN:
            iuen = semantic_unit[1]
            question = self.iuen_question(iuen[0])
            if iuen[0] == "none":
                cheat_sheet = {'no': (True, None), 'yes': (False, 0)}
            else:
                cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        elif semantic_tag == IUEN_v2:
            iuen = semantic_unit[1]
            question = self.iuen_question(iuen[0])
            cheat_sheet = {'yes': (True, None), 'no': (False, 0)}

        else:
            print("WARNING: Unknown semantic_tag %s" % semantic_tag)
            question = ""
            cheat_sheet = None

        # check nested WHERE/HAVING condition or IUEN != none
        closest_root_idx = helper_find_closest_bw(tag_seq, pointer - 1, tgt_name="root")
        if closest_root_idx == -1: # not found, not nested
            print('Question generated')
            return self.paraphrase_question(question), cheat_sheet
        else:
            root_tag = tag_seq[closest_root_idx][0]
            if root_tag == OUTSIDE: # IUEN != none
                print('Question generated')
                return self.paraphrase_question(question), cheat_sheet
            else:
                closest_end_nested_idx = helper_find_closest_bw(tag_seq, pointer - 1, tgt_name=END_NESTED)
                if closest_end_nested_idx != -1 and closest_end_nested_idx > closest_root_idx:
                    # outside the nested WHERE/HAVING condition
                    print('Question generated')
                    return self.paraphrase_question(question), cheat_sheet

                # nested WHERE/HAVING condition
                if root_tag == WHERE_ROOT_TERM:
                    (col,), op = tag_seq[closest_root_idx][1:3]
                    question = self.where_having_nested_question(col[1], col[0], op[0], question)
                elif root_tag == HAV_ROOT_TERM:
                    (col, agg), op = tag_seq[closest_root_idx][1:3]
                    question = self.where_having_nested_question(col[1], col[0], op[0], question,
                                                                 agg=agg[0] if agg[0] != 'none_agg' else None,
                                                                 bool_having=True)
                elif root_tag == HAV_ROOT_TERM_v2:
                    (col, agg, bool_distinct), op = tag_seq[closest_root_idx][1:3]
                    question = self.where_having_nested_question(col[1], col[0], op[0], question,
                                                                 agg=agg[0] if agg[0] != 'none_agg' else None,
                                                                 bool_having=True, bool_distinct=bool_distinct)
                else:
                    raise ValueError("Unexpected nested condition: tag_seq: {}\nPointer: {}, closest root: {}.".format(
                        tag_seq, pointer, tag_seq[closest_root_idx]
                    ))
                print('Question generated')
                return self.paraphrase_question(question), cheat_sheet

    def option_generation(self, cand_semantic_units, old_tag_seq, pointer):
        """
        Options generation.
        :param cand_semantic_units: a list of semantic units being the options.
        :param old_tag_seq: the original tag_seq, a sequence of semantic units.
        :param pointer: the pointer to the questioned semantic unit in old_tag_seq.
        :return: NL question, cheat_sheet = {choice idx: corresponding decision idx} (which will be used to simulate
                 user selections), the index for "none of the above".
        """
        semantic_tag = old_tag_seq[pointer][0]
        cheat_sheet = {}
        prefix, option_text = "", ""
        print('cand_semantic_units-->',repr(cand_semantic_units))
        print('old_tag_seq-->',repr(old_tag_seq))
        print('pointer-->',repr(pointer))
        
        if semantic_tag == SELECT_COL:
            prefix = "Please select any options below that I need to consider:\n"
            for idx, su in enumerate(cand_semantic_units):
                tab_col_item = su[1]
                option_text += "(%d) %s;\n" % (idx+1, self.agg_col_tab_description(tab_col_item[1], tab_col_item[0]))
                cheat_sheet[idx+1] = tab_col_item[-1] # col id

        elif semantic_tag == SELECT_AGG:
            prefix = "Please select any options below that I need to consider:\n"
            for idx, su in enumerate(cand_semantic_units):
                col, (agg, agg_idx) = su[1:3]
                if agg == "none_agg":
                    option_text += "(%d) the value of %s;\n" % (
                        idx + 1, self.agg_col_tab_description(col[1], col[0]))
                else:
                    option_text += "(%d) %s;\n" % (idx+1, self.agg_col_tab_description(col[1], col[0], agg=agg.lower()))
                cheat_sheet[idx+1] = (col[-1], agg_idx)

        elif semantic_tag == SELECT_AGG_v2:
            prefix = "Please select any options below that I need to consider:\n"
            for idx, su in enumerate(cand_semantic_units):
                col, (agg, agg_idx), bool_distinct = su[1:4]
                if agg == "none_agg":
                    option_text += "(%d) the value of %s;\n" % (
                        idx + 1, self.agg_col_tab_description(col[1], col[0]))
                else:
                    option_text += "(%d) %s;\n" % (
                        idx+1, self.agg_col_tab_description(col[1], col[0], agg=agg.lower(), bool_distinct=bool_distinct))
                cheat_sheet[idx+1] = (col[-1], agg_idx, bool_distinct)

        elif semantic_tag == WHERE_COL:
            prefix = "Please select any options below that I need to consider:\n"
            for idx, su in enumerate(cand_semantic_units):
                tab_col_item = su[1]
                option_text += "(%d) %s;\n" % (idx + 1, clean_words(self.agg_col_tab_description(tab_col_item[1], tab_col_item[0])))
                cheat_sheet[idx + 1] = tab_col_item[-1] # col id

        elif semantic_tag == WHERE_OP:
            prefix = "Please select any options below that I need to apply:\n"
            for idx, su in enumerate(cand_semantic_units):
                ((col,), (op_name, op_idx)) = su[1:3]
                condition_text = "%s %s a value" % (self.agg_col_tab_description(col[1], col[0]),
                                                    self.where_op.get(op_name, op_name))
                option_text += "(%d) %s;\n" % (idx+1, condition_text)
                cheat_sheet[idx+1] = (col[-1], op_idx) # (col id, op id)

        elif semantic_tag == WHERE_VAL:
            prefix = "Please select any options below that I need to apply:\n"
            for idx, su in enumerate(cand_semantic_units):
                ((col,), (op_name, op_idx), val_item) = su[1:4]
                condition_text = "%s %s \"%s\"" % (self.agg_col_tab_description(col[1], col[0]),
                                                    self.where_op.get(op_name, op_name), val_item[-1])
                option_text += "(%d) %s;\n" % (idx+1, condition_text)
                cheat_sheet[idx+1] = (col[-1], op_idx, val_item[-1]) # (col id, op id, val name)

        elif semantic_tag == GROUP_COL:
            prefix = "Please select any options from the following list:\n"
            for idx, su in enumerate(cand_semantic_units):
                tab_col_item = su[1]
                group_col_text = "The system needs to group %s" % (
                    self.group_by_agg_col_tab_description(tab_col_item[1], tab_col_item[0]))
                option_text += "(%d) %s;\n" % (idx+1, group_col_text)
                cheat_sheet[idx+1] = tab_col_item[-1] # col id

        elif semantic_tag == HAV_COL:
            prefix = "Following the last question, please select any options from below " \
                     "that I need to consider:\n"
            for idx, su in enumerate(cand_semantic_units):
                tab_col_item = su[1]
                option_text += "(%d) %s;\n" % (idx+1, self.agg_col_tab_description(
                    tab_col_item[1], tab_col_item[0], bool_having=True))
                cheat_sheet[idx + 1] = tab_col_item[-1] # col id

        elif semantic_tag == HAV_AGG:
            prefix = "Following the last question, please select one option from below " \
                     "that I need to consider:\n"
            for idx, su in enumerate(cand_semantic_units):
                col, agg = su[1:3]
                if agg[0] == "none_agg":
                    option_text += "(%d) the value of %s;\n" % (
                        idx+1, self.agg_col_tab_description(col[1], col[0], bool_having=True))
                else:
                    option_text += "(%d) %s;\n" % (idx+1, self.agg_col_tab_description(
                        col[1], col[0], agg=agg[0].lower(), bool_having=True))
                cheat_sheet[idx + 1] = (col[-1], agg[1]) # (col id, agg id)

        elif semantic_tag == HAV_AGG_v2:
            prefix = "Following the last question, please select any options from below " \
                     "that I need to consider:\n"
            for idx, su in enumerate(cand_semantic_units):
                col, agg, bool_distinct = su[1:4]
                if agg[0] == "none_agg":
                    option_text += "(%d) the value of %s;\n" % (
                        idx+1, self.agg_col_tab_description(col[1], col[0], bool_having=True))
                else:
                    option_text += "(%d) %s;\n" % (idx+1, self.agg_col_tab_description(
                        col[1], col[0], agg=agg[0].lower(), bool_having=True, bool_distinct=bool_distinct))
                cheat_sheet[idx + 1] = (col[-1], agg[1], bool_distinct)

        elif semantic_tag == HAV_OP:
            prefix = "Following the last question, please select any options from below " \
                     "that I need to apply:\n"
            for idx, su in enumerate(cand_semantic_units):
                (col, agg), op = su[1:3]
                condition_text = "%s %s a value" % (self.agg_col_tab_description(
                    col[1], col[0], agg=None if agg[0] == "none_agg" else agg[0], bool_having=True),
                                                    self.where_op.get(op[0], op[0]))
                option_text += "(%d) %s;\n" % (idx+1, condition_text)
                cheat_sheet[idx+1] = ((col[-1], agg[1]), op[1])

        elif semantic_tag == HAV_OP_v2:
            prefix = "Following the last question, please select any options from below " \
                     "that I need to apply:\n"
            for idx, su in enumerate(cand_semantic_units):
                (col, agg, bool_distinct), op = su[1:3]
                condition_text = "%s %s a value" % (self.agg_col_tab_description(
                    col[1], col[0], agg=None if agg[0] == "none_agg" else agg[0], bool_having=True,
                    bool_distinct=bool_distinct), self.where_op.get(op[0], op[0]))
                option_text += "(%d) %s;\n" % (idx+1, condition_text)
                cheat_sheet[idx+1] = ((col[-1], agg[1], bool_distinct), op[1])

        elif semantic_tag == ORDER_COL:
            prefix = "Please select any options from below, based on which" \
                     " I shall sort the results:\n"
            for idx, su in enumerate(cand_semantic_units):
                tab_col_item, = su[1]
                option_text += "(%d) %s;\n" % (idx + 1, self.agg_col_tab_description(
                    tab_col_item[1], tab_col_item[0]))
                cheat_sheet[idx + 1] = tab_col_item[-1] # col id

        elif semantic_tag == ORDER_AGG:
            prefix = "Please select ONE option from below, based on it " \
                     "I shall sort the results:\n"
            for idx, su in enumerate(cand_semantic_units):
                col, agg = su[1:3]
                if agg[0] == "none_agg":
                    option_text += "(%d) the value of %s;\n" % (
                        idx+1, self.agg_col_tab_description(col[1], col[0]))
                else:
                    option_text += "(%d) %s;\n" % (idx + 1, self.agg_col_tab_description(col[1], col[0],
                                                                                         agg=agg[0].lower()))
                cheat_sheet[idx + 1] = (col[-1], agg[1]) # (col id, agg id)

        elif semantic_tag == ORDER_AGG_v2:
            prefix = "Please select any options from below, based on which " \
                     "I shall sort the results:\n"
            for idx, su in enumerate(cand_semantic_units):
                col, agg, bool_distinct = su[1:4]
                if agg[0] == "none_agg":
                    option_text += "(%d) the value of %s;\n" % (
                        idx+1, self.agg_col_tab_description(col[1], col[0]))
                else:
                    option_text += "(%d) %s;\n" % (idx + 1, self.agg_col_tab_description(
                        col[1], col[0], agg=agg[0].lower(), bool_distinct=bool_distinct))
                cheat_sheet[idx + 1] = (col[-1], agg[1], bool_distinct)

        elif semantic_tag == ORDER_DESC_ASC_LIMIT:
            prefix = "Following the last question, please select ONE option from below:\n"
            for idx, su in enumerate(cand_semantic_units):
                (col, agg), desc_asc_limit = su[1:3]
                option_text += "(%d) %s;\n" % (idx+1, "The system should sort results " +
                                               self.desc_asc_limit[desc_asc_limit])
                cheat_sheet[idx + 1] = ((col[-1], agg[1]), desc_asc_limit)

        elif semantic_tag == IUEN:
            prefix = "Please select ONE option from the following list:\n"

            for idx, su in enumerate(cand_semantic_units):
                if su[1][0] == 'none':
                    iuen_text = "The system does NOT need to return information that satisfies a complicated situation " \
                                "as other options indicate"
                elif su[1][0] == 'except':
                    iuen_text = "The system needs to return information that satisfies some cases BUT NOT others, " \
                                "e.g., Find all airlines that have flights from airport 'CVO' BUT NOT from 'APG'"
                elif su[1][0] == 'union':
                    iuen_text = "The system needs to return information that satisfies either some cases OR others, " \
                                "e.g., What are the id and names of the countries which have more than 3 car makers " \
                                "OR produce the 'fiat' model?"
                else:
                    assert su[1][0] == 'intersect'
                    iuen_text = "The system needs to return information that satisfies BOTH some cases AND the others" \
                                " AT THE SAME TIME, e.g., Which district has BOTH stores with less than 3000 products " \
                                "AND stores with more than 10000 products?"
                option_text += "(%d) %s;\n" % (idx+1, iuen_text)
                cheat_sheet[idx + 1] = su[1][1] # iuen id

        elif semantic_tag == IUEN_v2:
            prefix = "Please select ONE option from the following list:\n"

            for idx, su in enumerate(cand_semantic_units):
                if su[1][0] == 'except':
                    iuen_text = "The system needs to return information that satisfies some cases BUT NOT others, " \
                                "e.g., Find all airlines that have flights from airport 'CVO' BUT NOT from 'APG'"
                elif su[1][0] == 'union':
                    iuen_text = "The system needs to return information that satisfies either some cases OR others, " \
                                "e.g., What are the id and names of the countries which have more than 3 car makers " \
                                "OR produce the 'fiat' model?"
                else:
                    assert su[1][0] == 'intersect'
                    iuen_text = "The system needs to return information that satisfies BOTH some cases AND the others" \
                                " AT THE SAME TIME, e.g., Which district has BOTH stores with less than 3000 products " \
                                "AND stores with more than 10000 products?"
                option_text += "(%d) %s;\n" % (idx + 1, iuen_text)
                cheat_sheet[idx + 1] = su[1][1]  # iuen id

        else:
            print("WARNING: Unknown semantic_tag %s" % semantic_tag)
            return "", cheat_sheet, -1

        if semantic_tag != IUEN:
            option_text += "(%d) None of the above options." % (len(cand_semantic_units) + 1)
            question = prefix + option_text
            return question, cheat_sheet, len(cheat_sheet) + 1
        else:
            question = prefix + option_text.strip()
            return question, cheat_sheet, -1

    def set_lang(self, lang):
        self.lang = lang

def clean_words(words):
    word=re.sub('[^a-zA-Z]+',' ',str(words))
    
    return word