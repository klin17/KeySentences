
import numpy as np
import os

def separate_highlights(filereader):
    highlights = []
    story = []
    was_highlight = False
    for line in filereader:
        stripped = line.strip()
        if len(stripped) > 0:
            # assumes that highlights always occur after the story
            if was_highlight:
                highlights.append(stripped)
            if len(highlights) == 0 and stripped != "@highlight":
                story.append(stripped)
            was_highlight = stripped == "@highlight"
    return story, highlights

if __name__ == "__main__":
    data_dir = "stories"
    out_dir = "unlabeled_stories"
    
    files = os.listdir(data_dir)
    for file in files:
        # create the file path
        path = os.path.join(data_dir, file)
        pre, ext = os.path.splitext(file)
        out_path = os.path.join(out_dir, pre + "_story.txt")
        with open(path, encoding="utf-8") as f:
            story, highlights = separate_highlights(f)
            full_story = " ".join(story)
            with open(out_path, "w", encoding="utf-8") as outf:
                outf.write(full_story)