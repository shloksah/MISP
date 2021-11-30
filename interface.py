import os
from flexx import flx, ui, event
import argparse

class Menu(flx.Widget):
    def init(self):
        global window
        self.colors = {'primary':'#2f2f2f','secondary':'#4c4c4c','tertiary':'#9f9f9f','text':'#ffffff','u1':'#34dd82','u2':'#7a91ff'}

        window.document.head.innerHTML = window.document.head.innerHTML + "<link href=\"https://fonts.googleapis.com/icon?family=Material+Icons+Sharp\" rel=\"stylesheet\">"
        with ui.VSplit(flex=1, style=f'background-color:{self.colors["primary"]};color:{self.colors["text"]};'):
            Head(self.colors)
            with ui.HSplit(flex=10):
                Chat(self.colors,flex=2)
                Data(self.colors,flex=3)


class Head(flx.Widget):
    def init(self, colors):
        self.colors = colors
        self.show_lng = False

        with flx.HBox(style=f'background-color:{self.colors["secondary"]};justify-content:flex-start;align-items:center;',flex=10):
            self.lang = ui.Button(text='language',style=f'background-color:{self.colors["secondary"]};color:{self.colors["text"]};',flex=1)
            google_btn(self.lang)

            self.vis_opt = DropdownMenu(
                self.colors,
                'visibility',
                ('Normal','Deuteranomaly','Protanomaly','Protanopia','Deuteranopia','Tritanopia','Tritanopia','Tritanomaly','Achromatopsia'),
                flex = 2
            )

            ui.Label(flex=8)

    @event.reaction('lang.pointer_click')
    def show_lang_opt(self, *events):
        if self.show_lng:
            print(events)
        else:
            print(events) 


class Chat(flx.Widget):
    def init(self, colors):
        self.colors = colors

        with ui.VSplit(style=f'background-color:{self.colors["secondary"]}'):
            self.messages = ui.VBox(flex=20,style='justify-content:flex-start;flex-direction:column;overflow-y:scroll;')

            with ui.HSplit(flex=1):
                self.input = ui.LineEdit(placeholder_text='Message...',flex=6,style=f'background-color:{self.colors["tertiary"]};color:{self.colors["text"]};')
                self.send_btn = ui.Button(text="keyboard_return",flex=1,style=f'background-color:{self.colors["tertiary"]};color:{self.colors["text"]};')
                google_btn(self.send_btn)
            

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


class DropdownMenu(flx.Widget):
    def init(self, colors, button, options):
        self.colors = colors

        with ui.VBox(style=f'background-color:{self.colors["secondary"]}') as self.dropdown:
            self.btn = ui.Button(text=button,style=f'background-color:{self.colors["tertiary"]};color:{self.colors["text"]};')
            google_btn(self.btn)
            self.content = ui.VBox(style=f'background-color:{self.colors["secondary"]};color:{self.colors["text"]};display:none;position:absolute;')

        opts = ''
        for option in options:
            opts += f'<ul>{option}</ul>'

        print(opts)
        print(self.content.node)

        self.content.node.innerHTML=opts
        print(self.content.node.innerHTML)

    @event.reaction('btn.pointer_click')
    def show_visability_opt(self, *events):
        self.toggle_display()

    def toggle_display(self):
        value = self.content.node.style.display

        if value == 'none':
            self.content.node.style.display = 'block'
        else:
            self.content.node.style.display = 'none'


def google_btn(node):
    node.node.className = node.node.className + ' material-icons-sharp'

# Runs the interface as a desktop app.
if __name__ == "__main__":
    # Build model
    # Build para

    app = flx.App(Menu)
    app.launch('app')
    flx.run()