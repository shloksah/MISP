import os
from flexx import flx, ui, event
import argparse

class Menu(flx.Widget):
    def init(self):
        self.colors = {'primary':'#2f2f2f','secondary':'#4c4c4c','tertiary':'#9f9f9f','text':'#ffffff','u1':'#34dd82','u2':'#7a91ff'}

        with ui.VSplit(flex=1, style=f'background-color:{self.colors["primary"]};color:{self.colors["text"]};'):
            Head(self.colors)
            with ui.HSplit(flex=10):
                Chat(self.colors,flex=2)
                Data(self.colors,flex=3)


class Head(flx.Widget):
    def init(self, colors):
        self.colors = colors

        with flx.HBox(style=f'background-color:{self.colors["secondary"]};justify-content:flex-start;align-items:center;'):
            ui.Label(text='Menu')
            ui.ComboBox(selected_key='Normal',style=f'background-color:{self.colors["tertiary"]};',options=('Normal','Deuteranomaly', 'Protanomaly', 'Protanopia', 'Deuteranopia', 'Tritanopia', 'Tritanopia', 'Tritanomaly', 'Achromatopsia'))
            ui.Label(text='Lang')


class Chat(flx.Widget):
    def init(self, colors):
        self.colors = colors

        with ui.VSplit(style=f'background-color:{self.colors["secondary"]}'):
            self.messages = ui.VBox(flex=20,style='justify-content:flex-start;flex-direction:column;overflow-y:scroll;')

            with ui.HSplit(flex=1):
                self.input = ui.LineEdit(placeholder_text='Message...',flex=6,style=f'background-color:{self.colors["tertiary"]};color:{self.colors["text"]};')
                self.send_btn = ui.Button(text="⎆",flex=1,style=f'background-color:{self.colors["tertiary"]};color:{self.colors["text"]};')
            

    @event.reaction('input.submit', 'send_btn.pointer_click')
    def send(self, *events):
        # Prevents empty lines.
        if self.input.text != '':
            self.textBubble(self.input.text,self.colors['u1'],'end')
            self.input.set_text('')
            self.input.set_placeholder_text('Thinking...')
            self.input.set_disabled(True)
            self.respond(self.input.text)
            self.messages.outernode.scrollTop = self.messages.outernode.scrollHeight-self.messages.outernode.clientHeight

    def respond(self, question):
        '''
        The function that sends the curent question to the model.

        Parameters
        ----------
        question: The question to ask the model.
        '''
        question = self.interaction(question)

        # Sends a response to the user.question
        self.textBubble(question,self.colors['u2'],'start')
        self.input.set_disabled(False)
        self.input.set_placeholder_text('Message...')

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

    def interaction(self, question):
        # model(question)
        return question + " Response..."


class Data(flx.Widget):
    def init(self, colors):
        self.colors = colors

        with flx.HSplit(style=f'background-color:{self.colors["secondary"]}'):
            pass


class BinarySelect(flx.Widget):
    def init(self):
        pass


class MultiSelect(flx.Widget):
    def init(self):
        pass


def interpret_args():
    """ Interprets the command line arguments, and returns a dictionary. """
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_bert', type=bool, default=False)
    parser.add_argument('--data_directory', type=str, default='processed_data')
    parser.add_argument('--output_embedding_size', type=int, default=300)
    parser.add_argument('--input_embedding_size', type=int, default=300)
    parser.add_argument('--freeze', type=bool, default=False)
    parser.add_argument('--encoder_state_size', type=int, default=300)
    parser.add_argument('--encoder_num_layers', type=int, default=1)
    parser.add_argument('--maximum_utterances', type=int, default=5)
    parser.add_argument("--bert_type_abb", type=str, help="Type of BERT model to load. e.g.) uS, uL, cS, cL, and mcS")
    args = parser.parse_args()

    return args

# Runs the interface as a desktop app.
if __name__ == "__main__":
    # params = interpret_args()
    # model = CustomModel(params)
    # model.load(os.path.join(os.getcwd(), "EditSQL/logs_clean/logs_spider_editsql_10p/model_best.pt"))

    # print(model)

    app = flx.App(Menu)
    flx.assets.add_shared_asset(asset_name='https://fonts.googleapis.com/icon?family=Material+Icons+Sharp')
    app.launch('app')
    flx.run()