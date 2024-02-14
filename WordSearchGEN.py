import random
import string
from typing import List, Tuple
from collections import namedtuple
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

Coordinate = namedtuple('Coordinate', ['x', 'y', 'direction'])

class WordSearch:
    def __init__(self, width: int, height: int, words: List[str], allow_horizontal=True, allow_vertical=True, allow_diagonal=True, allow_reverse=False) -> None:
        self.width = width
        self.height = height
        self.grid = [["" for _ in range(self.width)] for _ in range(self.height)]
        self.usage_count = [[0 for _ in range(self.width)] for _ in range(self.height)]
        self.placed_words = []
        self.words = self.sort_words_by_overlap(words)
        self.letter_positions = {}
        self.direction_count = {'H': 0, 'V': 0, 'DR': 0, 'DL': 0,'RH': 0, 'RV': 0, 'RDR': 0, 'RDL': 0}
        self.allow_horizontal = allow_horizontal
        self.allow_vertical = allow_vertical
        self.allow_diagonal = allow_diagonal
        self.allow_reverse = allow_reverse

    def calculate_overlap_score(self, word, all_words):
        score = 0
        word_set = set(word)
        for char in word_set:
            score += sum(char in other_word for other_word in all_words if other_word != word)
        return score

    def sort_words_by_overlap(self, words: List[str]):
        filtered_words = [word.upper() for word in words if len(word) <= max(self.width, self.height)]
        scores = {word: self.calculate_overlap_score(word, filtered_words) for word in filtered_words}
        return sorted(filtered_words, key=lambda word: scores[word], reverse=True)

    def place_word(self, word: str, coord: Coordinate):
        x, y, direction = coord

        reverse = direction.startswith("R")
        if reverse:
            direction = direction[1:]
            word = word[::-1]

        for i, char in enumerate(word):
            if direction == "H":
                self.grid[x][y + i] = char
                self.usage_count[x][y + i] += 1
                self.letter_positions.setdefault(char, set()).add((x, y + i))
            elif direction == "V":
                self.grid[x + i][y] = char
                self.usage_count[x + i][y] += 1
                self.letter_positions.setdefault(char, set()).add((x + i, y))
            elif direction == "DR":
                self.grid[x + i][y + i] = char
                self.usage_count[x + i][y + i] += 1
                self.letter_positions.setdefault(char, set()).add((x + i, y + i))
            elif direction == "DL":
                self.grid[x + i][y - i] = char
                self.usage_count[x + i][y - i] += 1
                self.letter_positions.setdefault(char, set()).add((x + i, y - i))
        self.placed_words.append((word, coord))

    def remove_word(self, word: str):
        for placed_word, coord in self.placed_words:
            if placed_word == word:
                x, y, direction = coord
                for i in range(len(word)):
                    if direction in ["H", "V"]:
                        cx, cy = x + (i if direction == "V" else 0), y + (i if direction == "H" else 0)
                    elif direction == "DR":
                        cx, cy = x + i, y + i
                    elif direction == "DL":
                        cx, cy = x + i, y - i

                    self.usage_count[cx][cy] -= 1
                    if self.usage_count[cx][cy] == 0:
                        self.grid[cx][cy] = ""

                self.placed_words.remove((word, coord))
                break

    def can_place_word(self, word: str, coord: Coordinate) -> bool:
        x, y, direction = coord
        reverse = direction.startswith("R")
        if reverse:
            direction = direction[1:]
            word = word[::-1]
        if direction == "H" and y + len(word) <= self.width:
            return all(self.grid[x][y + i] in ["", word[i]] for i in range(len(word)))
        elif direction == "V" and x + len(word) <= self.height:
            return all(self.grid[x + i][y] in ["", word[i]] for i in range(len(word)))
        elif direction == "DR" and x + len(word) <= self.height and y + len(word) <= self.width:
            return all(self.grid[x + i][y + i] in ["", word[i]] for i in range(len(word)))
        elif direction == "DL" and x + len(word) <= self.height and y - len(word) + 1 >= 0:
            return all(self.grid[x + i][y - i] in ["", word[i]] for i in range(len(word)))
        return False


    def suggest_coords(self, word: str):
        potential_coords = []
        directions = []
        if self.allow_horizontal:
            directions.append("H")
        if self.allow_vertical:
            directions.append("V")
        if self.allow_diagonal:
            directions.append("DR")
            directions.append("DL")
        if self.allow_reverse:
            directions = ["R" + d for d in directions] + directions  # Prepend 'R' for reverse

        random.shuffle(directions)

        for direction in directions:
            for x in range(self.height):
                for y in range(self.width):
                    coord = Coordinate(x, y, direction)
                    if self.can_place_word(word, coord):
                        overlap_score = self.calculate_overlap(coord, word)
                        potential_coords.append((coord, overlap_score))

        # Sort potential coordinates by overlap score in descending order
        potential_coords.sort(key=lambda x: x[1], reverse=True)

        # Return only the coordinates, not the scores
        return [coord for coord, _ in potential_coords]


    def calculate_overlap(self, coord: Coordinate, word: str) -> int:
        x, y, direction = coord
        score = 0
        for i, char in enumerate(word):
            if direction in ["H", "RH"] and y + i < self.width:
                score += 1 if self.grid[x][y + i] == char else 0
            elif direction in ["V", "RV"] and x + i < self.height:
                score += 1 if self.grid[x + i][y] == char else 0
            elif direction in ["DR", "RDR"] and x + i < self.height and y + i < self.width:
                score += 1 if self.grid[x + i][y + i] == char else 0
            elif direction in ["DL", "RDL"] and x + i < self.height and y - i >= 0:
                score += 1 if self.grid[x + i][y - i] == char else 0
        return score

    def backtrack(self, index: int, max_attempts: int, path: List[Tuple[str, Coordinate]]) -> bool:
        if index < 0:
            return False

        word, coord = path[index]
        self.remove_word(word)
        new_coords = self.suggest_coords(word)
        for new_coord in new_coords:
            if new_coord != coord and self.can_place_word(word, new_coord):
                self.place_word(word, new_coord)
                path[index] = (word, new_coord)
                if self.solve(index + 1, max_attempts, path):
                    return True
                self.remove_word(word)

        # Backtrack further
        return self.backtrack(index - 1, max_attempts, path)

    def solve(self, attempts=10000, max_iterations=5000):
        for attempt in range(attempts):
            print(f"Attempt {attempt + 1}")

            # Reset the state for a new attempt
            self.grid = [["" for _ in range(self.width)] for _ in range(self.height)]
            self.usage_count = [[0 for _ in range(self.width)] for _ in range(self.height)]
            self.placed_words = []
            unplaced_words = self.words.copy()
            stack = []
            path = []

            iteration_count = 0

            while iteration_count < max_iterations:
                iteration_count += 1

                # Check if all words have been placed
                if not unplaced_words:
                    # Successfully placed all words
                    return True

                # Always try to place the first word in the unplaced words list
                word = unplaced_words[0]

                # Prepare a new entry in the stack for the current word
                if not stack or stack[-1][0] != word:
                    potential_coords = self.suggest_coords(word)
                    stack.append((word, potential_coords, 0))

                _, potential_coords, coord_index = stack[-1]

                if coord_index < len(potential_coords):
                    coord = potential_coords[coord_index]
                    if self.can_place_word(word, coord):
                        self.place_word(word, coord)
                        path.append((word, coord))
                        unplaced_words.pop(0)  # Remove the placed word from unplaced words
                        stack[-1] = (word, potential_coords, coord_index + 1)
                    else:
                        stack[-1] = (word, potential_coords, coord_index + 1)
                else:
                    # Backtrack
                    if path:
                        prev_word, prev_coord = path.pop()
                        self.remove_word(prev_word)
                        unplaced_words.insert(0, prev_word)  # Add the word back to the front of unplaced words
                        
                        # Find next coordinate to try for the same word
                        prev_word_entry = stack.pop()
                        while stack and stack[-1][0] == prev_word:
                            prev_word_entry = stack.pop()

                        if prev_word_entry[2] < len(prev_word_entry[1]):
                            # There are still untried coordinates for this word
                            stack.append((prev_word, prev_word_entry[1], prev_word_entry[2]))
                        else:
                            random.shuffle(unplaced_words)  # Shuffle unplaced words
                    else:
                        # No more elements to backtrack, solution not possible in this attempt
                        break

            if not unplaced_words:
                return True  # Success

        return False  # All attempts failed


    def generate_word_search(self):
        if self.solve():
            self.fill_empty_spaces()
            return True
        else:
            return False

    def fill_empty_spaces(self):
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == "":
                    self.grid[y][x] = random.choice(string.ascii_uppercase)

    def print_highlighted_grid(self):
        highlighted_grid = [row[:] for row in self.grid]
        for word, coord in self.placed_words:
            x, y, direction = coord
            for i in range(len(word)):
                if direction == "H":
                    highlighted_grid[x][y + i] = self.grid[x][y + i].lower()
                elif direction == "V":
                    highlighted_grid[x + i][y] = self.grid[x + i][y].lower()
                elif direction == "DR":
                    highlighted_grid[x + i][y + i] = self.grid[x + i][y + i].lower()
                elif direction == "DL":
                    highlighted_grid[x + i][y - i] = self.grid[x + i][y - i].lower()

        for row in highlighted_grid:
            print(' '.join(row))
        print()

    def __repr__(self) -> str:
        return '\n'.join([''.join(row) for row in self.grid])

def visualize_word_search_with_lines(word_search):
    n_words = len(word_search.words)
    words_per_column = int(np.ceil(n_words / 3))
    word_columns = [word_search.words[i:i + words_per_column] for i in range(0, n_words, words_per_column)]
    word_columns += [''] * (3 - len(word_columns))

    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[word_search.height, 1, 0.2], hspace=0.4)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_aspect('equal')

    ax.set_xticks([])
    ax.set_yticks([])
    p=0

    for x in range(word_search.width):
        for y in range(word_search.height):
            ax.add_patch(plt.Rectangle((x, y), 1, 1, edgecolor='black', facecolor='none'))
            letter = word_search.grid[y][x]
            ax.text(x + 0.5, y + 0.5, letter, va='center', ha='center')

    for word, coord in word_search.placed_words:
        x, y, direction = coord
        reverse = direction.startswith("R")
        if reverse:
            direction = direction[1:]

        if direction == "H":
            start, end = (y + 0.5, x + 0.5), (y + len(word) - 0.5, x + 0.5)
            if reverse:
                start, end = end, start  # Swap start and end for reversed words
        elif direction == "V":
            start, end = (y + 0.5, x + 0.5), (y + 0.5, x + len(word) - 0.5)
            if reverse:
                start, end = end, start
        elif direction == "DR":
            start, end = (y + 0.5, x + 0.5), (y + len(word) - 0.5, x + len(word) - 0.5)
            if reverse:
                start, end = end, start
        elif direction == "DL":
            start, end = (y + 0.5, x + 0.5), (y - len(word) + 1.5, x + len(word) - 0.5)
            if reverse:
                start, end = end, start

        line = mlines.Line2D([start[0], end[0]], [start[1], end[1]], color='red', linewidth=2)
        ax.add_line(line)

        p=p+1

    plt.xlim(0, word_search.width)
    plt.ylim(0, word_search.height)
    plt.gca().invert_yaxis()

    ax_table = fig.add_subplot(gs[1, 0])
    ax_table.axis('off')
    table_data = list(map(list, zip(*word_columns)))
    table = ax_table.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[1/3]*3)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    print(str(p))
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('none')

    plt.show()

wordlist = [
    "Bilby", "Buffalo", "Bat", "Binturong", "Bull", "Haddock", "Kissingfish",
    "Puffer", "Mackerel", "Marlin", "Penguin", "Jay", "Lovebird", "Pigeon",
    "Macaw", "Tadpole", "Viper", "Tortoise", "Toad", "Dolphin"
]

word_search = WordSearch(11, 11, wordlist, allow_horizontal=True, allow_vertical=True, allow_diagonal=True, allow_reverse=True)

if word_search.generate_word_search():
    word_search.print_highlighted_grid()
    visualize_word_search_with_lines(word_search)
else:
    print("Failed to generate word search with the given words.")
