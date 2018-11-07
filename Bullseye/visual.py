width = 50

def print_title(title):
    msg = "\n========{title:=<{width}}========"
    print(msg.format(title=" "+title.upper()+" ",width=width))
def print_subtitle(subtitle):
    msg = "--------{title:-<{width}}--------"
    print(msg.format(title=" "+subtitle+" ",width=width))
    
if __name__ == "__main__":
    print_title("test")
    print_subtitle("test")
