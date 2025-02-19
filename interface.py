import os
from flexx import flx, ui, event
import argparse
from deep_translator import GoogleTranslator

# EditSQL imports
import random
import numpy as np
import torch
import json
from pscript.stubs import Math
from EditSQL.model.schema_interaction_model import SchemaInteractionATISModel
from EditSQL.data_util import atis_data
from EditSQL.question_gen import QuestionGenerator
from EditSQL.error_detector import ErrorDetectorProbability
from EditSQL.world_model import WorldModel
from EditSQL.agent import Agent
from EditSQL.environment import ErrorEvaluator, UserSim
from EditSQL.data_util.interaction import Interaction
from EditSQL.data_util.utterance import Utterance
from EditSQL.data_util.entities import NLtoSQLDict
from collections import defaultdict
import sqlite3
import time
import re
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
langs = {
    'Afrikaans':'af',
    'shqip':'sq',
    'ኣማርኛ':'am',
    'العربية':'ar',
    'Հայերէն':'hy',
    'آذربايجانجا ديلي':'az',
    'euskara':'eu',
    'Беларуская мова':'be',
    'বাংলা':'bn',
    'بۉسانسقى':'bs',
    'български':'bg',
    'català':'ca',
    'Sinugboanon':'ceb',
    '中文':'zh-CN',
    'corsu':'co',
    'Hrvatski':'hr',
    'čeština':'cs',
    'dansk':'da',
    'Nederlands':'nl',
    'English':'en',
    'Esperanto':'eo',
    'eesti keel':'et',
    'suomi':'fi',
    'français':'fr',
    'Frysk':'fy',
    'Galego':'gl',
    'ქართული':'ka',
    'Deutsch':'de',
    'ελληνικά':'el',
    'ગુજરાતી':'gu',
    'Kreyòl ayisyen':'ht',
    'حَوْسَ':'ha',
    'Española':'es'}
themes = {
    'Normal': {'primary':'#2f2f2f','secondary':'#4c4c4c','tertiary':'#9f9f9f','text':'#ffffff','u1':'#34dd82','u2':'#7a91ff'},
    'Deuteranomaly': {'primary':'#2f2f2f','secondary':'#4c4c4c','tertiary':'#9f9f9f','text':'#ffffff','u1':'#0000FF','u2':'#D500FF'},
    'Protanomaly': {'primary':'#2f2f2f','secondary':'#4c4c4c','tertiary':'#9f9f9f','text':'#ffffff','u1':'#FFB5EF','u2':'#59BFFF'},
    'Protanopia': {'primary':'#2f2f2f','secondary':'#4c4c4c','tertiary':'#9f9f9f','text':'#ffffff','u1':'#C4C800','u2':'#7F7FFF'},
    'Deuteranopia': {'primary':'#2f2f2f','secondary':'#4c4c4c','tertiary':'#9f9f9f','text':'#ffffff','u1':'#00E36A','u2':'#706DFF'},
    'Tritanopia': {'primary':'#2f2f2f','secondary':'#4c4c4c','tertiary':'#9f9f9f','text':'#ffffff','u1':'#00FF44','u2':'#FF6868'},
    'Tritanomaly': {'primary':'#2f2f2f','secondary':'#4c4c4c','tertiary':'#9f9f9f','text':'#ffffff','u1':'#FF27D0','u2':'#00CF75'},
    'Achromatopsia': {'primary':'#222222','secondary':'#666666','tertiary':'#aaaaaa','text':'#ffffff','u1':'#444444','u2':'#888888'}
}
coulors = themes['Normal']
lang='en'

def interpret_args():
    """ Interprets the command line arguments, and returns a dictionary. """
    parser = argparse.ArgumentParser()

    ### Data parameters
    parser.add_argument(
        '--raw_train_filename',
        type=str,
        default='../atis_data/data/resplit/processed/train_with_tables.pkl')
    parser.add_argument(
        '--raw_dev_filename',
        type=str,
        default='../atis_data/data/resplit/processed/dev_with_tables.pkl')
    parser.add_argument(
        '--raw_validation_filename',
        type=str,
        default='../atis_data/data/resplit/processed/valid_with_tables.pkl')
    parser.add_argument(
        '--raw_test_filename',
        type=str,
        default='../atis_data/data/resplit/processed/test_with_tables.pkl')

    parser.add_argument('--data_directory', type=str, default='processed_data')

    parser.add_argument('--processed_train_filename', type=str, default='train.pkl')
    parser.add_argument('--processed_dev_filename', type=str, default='dev.pkl')
    parser.add_argument('--processed_validation_filename', type=str, default='validation.pkl')
    parser.add_argument('--processed_test_filename', type=str, default='test.pkl')

    parser.add_argument('--database_schema_filename', type=str, default=None)
    parser.add_argument('--embedding_filename', type=str, default=None)

    parser.add_argument('--input_vocabulary_filename', type=str, default='input_vocabulary.pkl')
    parser.add_argument('--output_vocabulary_filename',
                        type=str,
                        default='output_vocabulary.pkl')

    parser.add_argument('--input_key', type=str, default='nl_with_dates')

    parser.add_argument('--anonymize', type=bool, default=False)
    parser.add_argument('--anonymization_scoring', type=bool, default=False)
    parser.add_argument('--use_snippets', type=bool, default=False)

    parser.add_argument('--use_previous_query', type=bool, default=False)
    parser.add_argument('--maximum_queries', type=int, default=1)
    parser.add_argument('--use_copy_switch', type=bool, default=False)
    parser.add_argument('--use_query_attention', type=bool, default=False)

    parser.add_argument('--use_utterance_attention', type=bool, default=False)

    parser.add_argument('--freeze', type=bool, default=False)
    parser.add_argument('--scheduler', type=bool, default=False)

    parser.add_argument('--use_bert', type=bool, default=False)
    parser.add_argument("--bert_type_abb", type=str, help="Type of BERT model to load. e.g.) uS, uL, cS, cL, and mcS")
    parser.add_argument("--bert_input_version", type=str, default='v1')
    parser.add_argument('--fine_tune_bert', type=bool, default=False)
    parser.add_argument('--lr_bert', default=1e-5, type=float, help='BERT model learning rate.')

    ### Debugging/logging parameters
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--deterministic', type=bool, default=False)
    parser.add_argument('--num_train', type=int, default=-1)

    parser.add_argument('--logfile', type=str, default='log.txt')
    parser.add_argument('--results_file', type=str, default='results.txt')

    ### Model architecture
    parser.add_argument('--input_embedding_size', type=int, default=300)
    parser.add_argument('--output_embedding_size', type=int, default=300)

    parser.add_argument('--encoder_state_size', type=int, default=300)
    parser.add_argument('--decoder_state_size', type=int, default=300)

    parser.add_argument('--encoder_num_layers', type=int, default=1)
    parser.add_argument('--decoder_num_layers', type=int, default=2)
    parser.add_argument('--snippet_num_layers', type=int, default=1)

    parser.add_argument('--maximum_utterances', type=int, default=5)
    parser.add_argument('--state_positional_embeddings', type=bool, default=False)
    parser.add_argument('--positional_embedding_size', type=int, default=50)

    parser.add_argument('--snippet_age_embedding', type=bool, default=False)
    parser.add_argument('--snippet_age_embedding_size', type=int, default=64)
    parser.add_argument('--max_snippet_age_embedding', type=int, default=4)
    parser.add_argument('--previous_decoder_snippet_encoding', type=bool, default=False)

    parser.add_argument('--discourse_level_lstm', type=bool, default=False)

    parser.add_argument('--use_schema_attention', type=bool, default=False)
    parser.add_argument('--use_encoder_attention', type=bool, default=False)

    parser.add_argument('--use_schema_encoder', type=bool, default=False)
    parser.add_argument('--use_schema_self_attention', type=bool, default=False)
    parser.add_argument('--use_schema_encoder_2', type=bool, default=False)

    ### Training parameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_maximum_sql_length', type=int, default=200)
    parser.add_argument('--train_evaluation_size', type=int, default=100)

    parser.add_argument('--dropout_amount', type=float, default=0.5)

    parser.add_argument('--initial_patience', type=float, default=10.)
    parser.add_argument('--patience_ratio', type=float, default=1.01)

    parser.add_argument('--initial_learning_rate', type=float, default=0.001)
    parser.add_argument('--learning_rate_ratio', type=float, default=0.8)

    parser.add_argument('--interaction_level', type=bool, default=True)
    parser.add_argument('--reweight_batch', type=bool, default=False)

    ### Setting
    # parser.add_argument('--train', type=bool, default=False)
    parser.add_argument('--train', type=int, choices=[0,1], default=0)
    parser.add_argument('--debug', type=bool, default=False)

    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--attention', type=bool, default=False)
    parser.add_argument('--enable_testing', type=bool, default=False)
    parser.add_argument('--use_predicted_queries', type=bool, default=False)
    parser.add_argument('--evaluate_split', type=str, default='dev')
    parser.add_argument('--evaluate_with_gold_forcing', type=bool, default=False)
    parser.add_argument('--eval_maximum_sql_length', type=int, default=1000)
    parser.add_argument('--results_note', type=str, default='')
    parser.add_argument('--compute_metrics', type=bool, default=False)

    parser.add_argument('--reference_results', type=str, default='')

    parser.add_argument('--interactive', type=bool, default=False)

    parser.add_argument('--database_username', type=str, default="aviarmy")
    parser.add_argument('--database_password', type=str, default="aviarmy")
    parser.add_argument('--database_timeout', type=int, default=2)

    # interaction params - Ziyu
    parser.add_argument('--job', default='test_w_interaction', choices=['test_w_interaction', 'online_learning'],
                        help='Set the job. For parser pretraining, see other scripts.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--raw_data_directory', type=str, help='The data directory of the raw spider data.')

    parser.add_argument('--num_options', type=str, default='3', help='[INTERACTION] Number of options.')
    parser.add_argument('--user', type=str, default='sim', choices=['sim', 'gold_sim', 'real'],
                        help='[INTERACTION] User type.')
    parser.add_argument('--err_detector', type=str, default='any',
                        help='[INTERACTION] The error detector: '
                             '(1) prob=x for using policy probability threshold;'
                             '(2) stddev=x for using Bayesian dropout threshold (need to set --dropout and --passes);'
                             '(3) any for querying about every policy action;'
                             '(4) perfect for using a simulated perfect detector.')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='[INTERACTION] Dropout rate for Bayesian dropout-based uncertainty analysis. '
                             'This does NOT change the dropout rate in training.')
    parser.add_argument('--passes', type=int, default=1,
                        help='[INTERACTION] Number of decoding passes for Bayesian dropout-based uncertainty analysis.')
    parser.add_argument('--friendly_agent', type=int, default=0, choices=[0, 1],
                        help='[INTERACTION] If 1, the agent will not trigger further interactions '
                             'if any wrong decision is not resolved during parsing.')
    parser.add_argument('--ask_structure', type=int, default=0, choices=[0, 1],
                        help='[INTERACTION] Set to True to allow questions about query structure '
                             '(WHERE/GROUP_COL, ORDER/HAV_AGG_v2) in NL.')
    parser.add_argument('--output_path', type=str, default='temp', help='[INTERACTION] Where to save outputs.')

    # online learning
    parser.add_argument('--setting', type=str, default='', choices=['online_pretrain_10p', 'full_train'],
                        help='Model setting; checkpoints will be loaded accordingly.')
    parser.add_argument('--supervision', type=str, default='full_expert',
                        choices=['full_expert', 'misp_neil', 'misp_neil_perfect', 'misp_neil_pos',
                                 'bin_feedback', 'bin_feedback_expert',
                                 'self_train', 'self_train_0.5'],
                        help='[LEARNING] Online learning supervision based on different algorithms.')
    parser.add_argument('--data_seed', type=int, choices=[0, 10, 100],
                        help='[LEARNING] Seed for online learning data.')
    parser.add_argument('--start_iter', type=int, default=0, help='[LEARNING] Starting iteration in online learing.')
    parser.add_argument('--end_iter', type=int, default=-1, help='[LEARNING] Ending iteration in online learing.')
    parser.add_argument('--update_iter', type=int, default=1000,
                        help='[LEARNING] Number of iterations per parser update.')

    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)

    if not (args.train or args.evaluate or args.interactive or args.attention):
        raise ValueError('You need to be training or evaluating')
    if args.enable_testing and not args.evaluate:
        raise ValueError('You should evaluate the model if enabling testing')

    # Seeds for random number generation
    print("## seed: %d" % args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    return args

params = interpret_args()

# Prepare the dataset into the proper form.
data = atis_data.ATISDataset(params)

question_generator = QuestionGenerator(bool_structure_question=True, lang='en')

# model loading
model = SchemaInteractionATISModel(
    params,
    data.input_vocabulary,
    data.output_vocabulary,
    data.output_vocabulary_schema,
    None)
model.load(os.path.join("EditSQL/logs_clean/logs_spider_editsql_10p", "model_best.pt"))
model = model.to(device)

error_detector = ErrorDetectorProbability(0.995)
world_model = WorldModel(model, 3, None, 1, 0.0,
                        bool_structure_question=True)

raw_valid_examples = json.load(open(os.path.join(params.raw_data_directory, "dev_reordered.json")))

class Menu(flx.Widget):
    def init(self, model, colors):
        with ui.VSplit(flex=1, style=f'background-color:{colors["primary"]};color:{colors["text"]};'):
            Head(model,colors,style='overflow:visible;',flex=1)
            with ui.HSplit(flex=10):
                Chat(model,colors,flex=2)
                Data(model,colors,flex=2)


class Head(flx.Widget):
    def init(self, model, colors):
        global window
        self.model = model
        self.colors = colors

        window.document.head.innerHTML = window.document.head.innerHTML + "<link href=\"https://fonts.googleapis.com/icon?family=Material+Icons+Sharp\" rel=\"stylesheet\">"
        with flx.HBox(style=f'background-color:{self.colors["secondary"]};justify-content:flex-start;align-items:center;overflow:visible;',flex=10):   
            DropdownMenu(
                'language',
                langs.keys(),
                self.set_lang,
                self.colors
            )

            DropdownMenu(
                'visibility',
                ('Normal','Deuteranomaly','Protanomaly','Protanopia','Deuteranopia','Tritanopia','Tritanomaly','Achromatopsia'),
                self.set_theme,
                self.colors
            )

            ui.Label(flex=8)
    
    def set_lang(self):
        global window
        nodes = window.document.querySelectorAll( ":hover" )
        self.model.set_lang(langs[nodes[nodes.length - 1].innerText])

    def set_theme(self):
        global window
        nodes = window.document.querySelectorAll( ":hover" )
        self.model.set_theme(nodes[nodes.length - 1].innerText)
        window.location.reload()


class Chat(flx.Widget):
    def init(self, model, colors):
        self.model = model
        self.colors = colors

        with ui.VSplit(style=f'background-color:{colors["secondary"]}'):
            self.messages = ui.VBox(flex=20,style='justify-content:flex-start;flex-direction:column;overflow-y:scroll;')

            with ui.HSplit(flex=1):
                self.input = ui.LineEdit(placeholder_text='',flex=6,style=f'background-color:{colors["tertiary"]};color:{colors["text"]};')
                self.send_btn = ui.Button(text="keyboard_return",flex=1,style=f'background-color:{colors["tertiary"]};color:{colors["text"]};')
                self.send_btn.node.className = self.send_btn.node.className + ' material-icons-sharp'

    @event.reaction('model.question')    
    def type_q(self):
        self.input.set_text(self.model.question)

        if "\t" in self.model.question:
            self.send(None)

    @event.reaction('input.submit', 'send_btn.pointer_click')
    def send(self, *events):
        # Prevents empty lines.
        if self.input.text != '':
            self.textBubble(self.input.text,self.colors['u1'],'end')
            self.input.set_text('')
        else:
            # Runs an interaction.
            self.model.interaction()

    @event.reaction('model.response')
    def interaction(self):
        # Sends a response to the user.question
        self.textBubble(self.model.response,self.colors['u2'],'start')

    def textBubble(self, text, color, side):
        """
        Creates a text bubble for the text sent to the device.
        
        Parameters
        ----------
        text: The text to have the user send.
        color: The color of the text.
        side: Which side to display the text on. [start, end]
        """
        global window
        outernode = window.document.createElement('div')
        node = window.document.createElement('p')
        node.innerText = text
        node.style = f'background-color:{color};border-radius:0.5rem;padding:0.5rem;margin:0.25rem;'
        outernode.style = f'position:relative;display:inline-flex;justify-content:{side};'
        outernode.appendChild(node)
        self.messages.outernode.appendChild(outernode)
        self.messages.outernode.scrollTop = self.messages.outernode.scrollHeight-self.messages.outernode.clientHeight
                            

class Data(flx.CanvasWidget):
    def init(self, model, colors):
        super().init()
        self.ctx = self.node.getContext('2d')
        self.colors = colors
        self.model = model
        self.time = 0

        # Starts animation loop.
        self.update()

    def update(self):
        global window
        ctx = self.ctx

        # Variables
        width = self.ctx.canvas.width
        height = self.ctx.canvas.height
        center = (width / 2, height / 2)
        radius = center[1]/1.5
        num_circles = 50
        sml_radius = radius/num_circles

        if self.model.mode == 'breath':
            # Draws background.
            ctx.fillStyle = self.colors["secondary"]
            ctx.fillRect(0, 0, width, height)

            # Breathing mode.
            ctx.fillStyle = self.colors["u2"]
            for x in range(-int(num_circles/2), num_circles-int(num_circles/2)):
                for y in range(-int(num_circles/2), num_circles-int(num_circles/2)):
                    if Math.sqrt((2*x*sml_radius)**2 + (2*y*sml_radius)**2) < radius:
                        if self.time <= 2000:
                            ctx.beginPath()
                            ctx.arc(center[0] + 2*x*sml_radius, center[1] + 2*y*sml_radius, (sml_radius * (self.time / 2000)) / 1.2, 0, 6.283185)
                            ctx.fill()
                        else:
                            ctx.beginPath()
                            ctx.arc(center[0] + 2*x*sml_radius, center[1] + 2*y*sml_radius, (sml_radius * (1 - (self.time - 2000) / 2000)) / 1.2, 0, 6.283185)
                            ctx.fill()
        elif self.model.mode == 'calc' and self.time % 60 == 0:
            # Draws background.
            ctx.fillStyle = self.colors["secondary"]
            ctx.fillRect(0, 0, width, height)

            # Calculate mode.
            ctx.fillStyle = self.colors["u2"]
            for x in range(-int(num_circles/2), num_circles-int(num_circles/2)):
                for y in range(-int(num_circles/2), num_circles-int(num_circles/2)):
                    if Math.sqrt((2*x*sml_radius)**2 + (2*y*sml_radius)**2) < radius:
                        ctx.beginPath()
                        ctx.arc(center[0] + 2*x*sml_radius, center[1] + 2*y*sml_radius, (sml_radius * Math.random()) / 1.2, 0, 6.283185)
                        ctx.fill()
        elif self.model.mode == 'data':
            # Draws background.
            ctx.fillStyle = self.colors["secondary"]
            ctx.fillRect(0, 0, width, height)

            # Text settings
            ctx.fillStyle = self.colors["text"]
            ctx.strokeStyle = self.colors["primary"]
            ctx.font = '24px serif'
            ctx.textAlign = 'left'
            ctx.lineWidth = 4
            num_rows = len(self.model.table)
            num_cols = len(self.model.table[0])
            max_width = [0]*num_cols
            max_height = 34

            # Caculates the size of the table
            for row in self.model.table:
                for i, column in enumerate(row):
                    metrics = ctx.measureText(column)

                    if metrics.width + 16 > max_width[i]:
                        max_width[i] = metrics.width + 16

            # Finds table settings.
            table_width = sum(max_width)
            table_height = max_height * num_rows
            cur_width = 0

            # Draws the table.
            for r in range(num_rows-1):
                ctx.beginPath()
                ctx.moveTo(center[0] - table_width / 2, center[1] + (table_height / 2) - (max_height * (r + 1)))  
                ctx.lineTo(center[0] + table_width / 2, center[1] + (table_height / 2) - (max_height * (r + 1)))
                ctx.stroke()
            for c in range(num_cols-1):
                cur_width += max_width[num_cols-1-c]
                ctx.beginPath()       
                ctx.moveTo(center[0] + (table_width / 2) - (cur_width), center[1] + table_height / 2)  
                ctx.lineTo(center[0] + (table_width / 2) - (cur_width), center[1] - table_height / 2)
                ctx.stroke()

            # Draws the text
            for r in range(num_rows):
                cur_width = 0
                for c in range(num_cols):
                    ctx.fillText(self.model.table[r][c], center[0] - (table_width / 2) + cur_width + 8, center[1] - (table_height / 2) + (max_height * (r+1)) - 10)
                    cur_width += max_width[c]

        # Loops window.
        self.time += 10
        self.time %= 4000
        window.setTimeout(self.update, 10)


class DropdownMenu(flx.Widget):
    def init(self, button, options, func, colors):
        self.option = ""
        self.button = button
        self.options = options
        self.func = func
        self.colors = colors

    def _render_dom(self):
        return flx.create_element('div', {'style': 'position:relative;display:inline;overflow:visible;'},
            flx.create_element('button', {
                'style': f'background-color:{self.colors["secondary"]};color:{self.colors["text"]};',
                'class': 'flx-Button flx-BaseButton flx-Widget material-icons-sharp',
                'onclick': self.toggle_display
            }, self.button),
            flx.create_element('div', {
                'style': f"""
                    position:absolute;display:none;z-index:5;background-color:{self.colors["secondary"]};
                    border-radius:0.5rem;box-shadow: 2px 2px 4px 2px rgba(0, 0, 0, 0.2);
                    cursor: pointer;overflow-y:scroll;max-height:50vh;""",
                'onmouseleave': self.hide_display,
            },
                [flx.create_element('li', {
                    'style': "list-style:none;padding:0.5rem;border-radius:0.5rem;",
                    'onmouseover': self.hover_on,
                    'onmouseleave': self.hover_off,
                    'onclick': self.click
                }, option) for option in self.options]
            )
        )

    def click(self):
        self.func()
        self.hide_display()

    def hover_on(self):
        global window
        nodes = window.document.querySelectorAll( ":hover" )
        nodes[nodes.length - 1].style.backgroundColor = self.colors["primary"]

    def hover_off(self):
        global window
        nodes = window.document.querySelectorAll( ":hover" )

        for i in range(self.outernode.lastChild.children.length):
            child = self.outernode.lastChild.children[i]
            if child != nodes[nodes.length - 1]:
                child.style.backgroundColor = self.colors['secondary']
                
    def hide_display(self):
        self.outernode.lastChild.style.display = 'none'

    def toggle_display(self):
        value = self.outernode.lastChild.style.display

        if value == 'none':
            self.outernode.lastChild.style.display = 'block'
        else:
            self.outernode.lastChild.style.display = 'none'


class Edit_SQL(flx.PyComponent):
    question = flx.StringProp('', settable=True)
    response = flx.StringProp('', settable=True)
    mode = flx.StringProp('breath', settable=True)
    table = flx.ListProp([], settable=True)

    def init(self):
        # The interface.
        self.agent = Agent(world_model, error_detector, question_generator,
                    bool_mistake_exit=False,
                    bool_structure_question=True,
                    set_text=self.set_text)

        # environment setup: user simulator
        error_evaluator = ErrorEvaluator()
        self.user = UserSim(error_evaluator, set_text=self.set_text, lang='en', bool_structure_question=params.ask_structure)

        # Shows client interface.
        Menu(self, coulors)

    def extract_clause_asterisk(self, g_sql_toks):
        """
        This function extracts {clause keyword: tab_col_item with asterisk (*)}.
        Keywords include: SELECT/HAV/ORDER_AGG_v2.
        A tab_col_item lookds like "*" or "tab_name.*".

        The output will be used to simulate user evaluation and selections.
        The motivation is that the structured "g_sql" does not contain tab_name for *, so the simulator cannot decide the
        right decision precisely.
        :param g_sql_toks: the preprocessed gold sql tokens from EditSQL.
        :return: A dict of {clause keyword: tab_col_item with asterisk (*)}.
        """
        kw2item = defaultdict(list)

        keyword = None
        for tok in g_sql_toks:
            if tok in {'select', 'having', 'order_by', 'where', 'group_by'}:
                keyword = tok
            elif keyword in {'select', 'having', 'order_by'} and (tok == "*" or re.findall("\.\*", tok)):
                kw2item[keyword].append(tok)

        kw2item = dict(kw2item)
        for kw, item in kw2item.items():
            try:
                assert len(item) <= 1
            except:
                print("\nException in clause asterisk extraction:\ng_sql_toks: {}\nkw: {}, item: {}\n".format(
                    g_sql_toks, kw, item))
            kw2item[kw] = item[0]

        return kw2item

    def set_text(self, target, value):
        if target == 'q':
            q = ''
            for l in value:
                q += l
                self.set_question(q.strip())
                time.sleep(random.uniform(0.1, 0.05))
            self.set_question(q.strip() + '\t')

            # Shows AI thinking
            self.set_mode('calc')
        else:
            if self.response == value:
                self.set_response(value + '\n')
            else:
                self.set_response(value)

            # Shows AI chilling
            self.set_mode('breath')
        time.sleep(1)

    @flx.action
    def set_theme(self, theme):
        global coulors
        coulors = themes[theme]

    @flx.action
    def set_lang(self, lng):
        global lang
        self.user.set_lang(lng)
        question_generator.set_lang(lng)
        lang=lng

    @flx.action
    def interaction(self):
        """ Evaluates a sample of interactions. """
        # Gets the data
        self.reorganized_data = list(zip(raw_valid_examples, data.get_all_interactions(data.valid_data)))
        random.shuffle(self.reorganized_data)
        (raw_example, example) = self.reorganized_data[1]
        # Sets the question
        question = ' '.join(example.interaction.utterances[0].original_input_seq)
        
        question_op = GoogleTranslator(source='auto', target=lang).translate(question)
        self.set_text('q', question_op)

        max_generation_length = 100
        count_exception = 0
        with torch.no_grad():
            input_item = self.agent.world_model.semparser.spider_single_turn_encoding(
                example, max_generation_length)

            true_sql = example.interaction.utterances[0].original_gold_query
            g_sql = raw_example['sql']
            g_sql["column_names_surface_form_to_id"] = input_item[-1].column_names_surface_form_to_id
            g_sql["base_vocab"] = self.agent.world_model.vocab
            g_sql["extracted_clause_asterisk"] = self.extract_clause_asterisk(true_sql)

            try:
                hyp = self.agent.world_model.decode(input_item, bool_verbal=False, dec_beam_size=1)[0]
            except Exception: # tag_seq generation exception - e.g., when its syntax is wrong
                count_exception += 1
                final_encoder_state, encoder_states, schema_states, max_generation_length, snippets, input_sequence, \
                    previous_queries, previous_query_states, input_schema = input_item
                prediction = self.agent.world_model.semparser.decoder(
                    final_encoder_state,
                    encoder_states,
                    schema_states,
                    max_generation_length,
                    snippets=snippets,
                    input_sequence=input_sequence,
                    previous_queries=previous_queries,
                    previous_query_states=previous_query_states,
                    input_schema=input_schema,
                    dropout_amount=0.0)
                sequence = prediction.sequence
                probability = prediction.probability
            else:
                try:
                    new_hyp, bool_exit, question = self.agent.interactive_parsing_session(self.user, input_item, g_sql, hyp,
                                                                            bool_verbal=False)
                    if bool_exit:
                        count_exit += 1
                except Exception:
                    count_exception += 1
                    bool_exit = False
                else:
                    hyp = new_hyp

                sequence = hyp.sql
                probability = np.exp(hyp.logprob)

            sys.stdout.flush()

        # Connect to database.
        try:
            conn = sqlite3.connect(os.path.join(os.getcwd(), f'EditSQL/data_clean/database/{raw_example["db_id"]}/{raw_example["db_id"]}.sqlite'))
        except:
            quit()

        # Gets the data
        cur = conn.cursor()
        cur.execute(raw_example['query'])

        # Prints the data
        rows = cur.fetchall()

        if len(rows) == 0:
            no_res = 'Unfortanatly, there were no resluts.'
            no_res = GoogleTranslator(source='auto', target=lang).translate(no_res)
            self.set_text('r', no_res)
            self.set_mode('breath')
        elif len(rows) == 1:
            line = str(cur.description[0][0]) + ': ' + str(rows[0])
            line = line.translate({ord(c): None for c in '@#^&*-()'})
            line = line.translate({ord(c): ' ' for c in '_'})
            line = GoogleTranslator(source='auto', target=lang).translate(line)
            self.set_text('r', line)
            self.set_mode('breath')
        else:
            table = []
            table.append(tuple([GoogleTranslator(source='auto', target=lang).translate(description[0].translate({ord(c): ' ' for c in '_'}).translate({ord(c): None for c in '@#^&*-()'}).title()) for description in cur.description]))
            for row in rows:
                table.append(row)
            line = 'Displaying data on the right.'
            line = GoogleTranslator(source='auto', target=lang).translate(line)
            self.set_text('r', line)
            self.set_table(table)
            self.set_mode('data')

        
# Runs the interface as a desktop app.
if __name__ == "__main__":
    # App
    app = flx.App(Edit_SQL)
    app.launch('browser')
    flx.run()