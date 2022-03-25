import tkinter as tk

pressed = False


class CanvasInFrame(tk.Frame):
    def __init__(self, width, height, bg='gray30'):
        self.root = tk.Tk()
        tk.Frame.__init__(self, self.root)
        self.canvas = tk.Canvas(self, width=width, height=height, background=bg)
        self.xsb = tk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.ysb = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.ysb.set, xscrollcommand=self.xsb.set)
        self.canvas.configure(scrollregion=(0,0,height*2,width*2))

        self.xsb.grid(row=1, column=0, sticky="ew")
        self.ysb.grid(row=0, column=1, sticky="ns")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # This is what enables using the mouse:
        self.canvas.bind("<ButtonPress-1>", self.move_start)
        self.canvas.bind("<B1-Motion>", self.move_move)

        self.canvas.bind("<ButtonPress-2>", self.pressed2)
        self.canvas.bind("<Motion>", self.move_move2)

        # windows scroll
        self.canvas.bind("<MouseWheel>",self.zoomer)
        # Hack to make zoom work on Windows
        self.root.bind_all("<MouseWheel>",self.zoomer)

        self.pack(fill="both", expand=True)

    # move
    def move_start(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def move_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    # move
    def pressed2(self, event):
        global pressed
        pressed = not pressed
        self.canvas.scan_mark(event.x, event.y)

    def move_move2(self, event):
        if pressed:
            self.canvas.scan_dragto(event.x, event.y, gain=1)

    # windows zoom
    def zoomer(self,event):
        if event.delta > 0:
            self.canvas.scale("all", event.x, event.y, 1.1, 1.1)
        elif event.delta < 0:
            self.canvas.scale("all", event.x, event.y, 0.9, 0.9)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))