class Rect:
    def __init__(self, left, top, right, bottom):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def expand(self, x, y): # Expands the rectangle
        self.left = min([self.left, x]) # Calculates the left of rectangle
        self.top = min([self.top, y]) # Calculates the top of rectangle
        self.right = max([self.right, x]) # Calculates the right of rectangle
        self.bottom = max([self.bottom, y]) # Calculates the bottom of rectangle

    def get_width(self):
        return self.right - self.left + 1 # Calculates the width of rectangle

    def get_height(self):
        return self.bottom - self.top + 1 # Calculates the heigt of rectangle

    def get_area(self):
        return self.get_width() * self.get_height()

def is_intersect(rt1, rt2):
    if rt1.right < rt2.left or rt1.left > rt2.right:
        return False
    if rt1.bottom < rt2.top or rt1.top > rt2.bottom:
        return False
    return True

def is_same(rt1, rt2):
    # Checks whether the rectangles are same
    if rt1.left == rt2.left and rt1.top == rt2.top and rt1.right == rt2.right and rt1.bottom == rt2.bottom:
        return True
    return False