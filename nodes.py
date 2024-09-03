import os.path
import os
import random
import inspect
import pathlib
import re
import time
import glob
import yaml

ALL_KEY = 'all yaml files'


# Defines a function get_index that takes a list and item
def get_index(items, item):
    # Tries to lookup item's index in items using .index
    try:
        return items.index(item)

    # If there is an exception, return None
    except Exception:
        return None


# Defines a function parse_tag to clean up a tag string
def parse_tag(tag):
    # Replace underscores and angle brackets
    tag = tag.replace("__", "").replace('<', '').replace('>', '')

    # Strip whitespace
    tag = tag.strip()

    # Return cleaned up tag
    return tag


# Defines a function to read lines from a file
def read_file_lines(file):
    # Read file lines
    f_lines = file.read().splitlines()

    # Initialize empty list for cleaned lines
    lines = []

    # Loop through each line
    for line in f_lines:

        # Strip whitespace
        line = line.strip()

        # Skip line if it's empty or starts with #
        if line and not line.startswith('#'):
            # Append cleaned line
            lines.append(line)

    # Return list of cleaned lines
    return lines


# Wildcards class
class TagLoader:
    # Class variables
    files = []
    wildcard_location = os.path.join(
        pathlib.Path(inspect.getfile(lambda: None)).parent,
        "wildcards"
    )
    loaded_tags = {}
    missing_tags = set()

    # Method to load tags from file
    def load_tags(self, file_path, verbose=False, cache_files=False):

        # Check cache
        if cache_files and self.loaded_tags.get(file_path):
            return self.loaded_tags.get(file_path)

        # Get file paths
        txt_file_path = os.path.join(self.wildcard_location, f'{file_path}.txt')
        yaml_file_path = os.path.join(self.wildcard_location, f'{file_path}.yaml')

        # Get key
        if (file_path == ALL_KEY):
            key = ALL_KEY
        else:
            key = file_path.lower()

        # Load from .txt file
        if self.wildcard_location and os.path.isfile(txt_file_path):
            with open(txt_file_path, encoding="utf8") as file:
                self.files.append(f"{file_path}.txt")
                self.loaded_tags[key] = read_file_lines(file)

        # Load all .yaml files
        if key is ALL_KEY and self.wildcard_location:
            files = glob.glob(os.path.join(self.wildcard_location, '**/*.yaml'), recursive=True)
            output = {}
            for file in files:
                with open(file, encoding="utf8") as file:
                    self.files.append(f"{file_path}.yaml")
                    path = os.path.relpath(file.name)
                    try:
                        data = yaml.safe_load(file)
                        # Process YAML data
                        if not isinstance(data, dict) and verbose:
                            print(f'Warning: Missing contents in {path}')
                            continue

                        for item in data:
                            if (hasattr(output, item) and verbose):
                                print(f'Warning: Duplicate key "{item}" in {path}')
                            if data[item] and 'Tags' in data[item]:
                                if not isinstance(data[item]['Tags'], list) and verbose:
                                    print(
                                        f'Warning: No tags found in at item "{item}" (add at least one tag to it) in {path}')
                                    continue
                                output[item] = {
                                    x.lower().strip()
                                    for i, x in enumerate(data[item]['Tags'])
                                }
                            else:
                                if verbose: print(f'Warning: No "Tags" section found in at item "{item}" in {path}')
                    except yaml.YAMLError as exc:
                        print(exc)
            self.loaded_tags[key] = output

        # Load single .yaml file
        if self.wildcard_location and os.path.isfile(yaml_file_path):
            with open(yaml_file_path, encoding="utf8") as file:
                self.files.append(f"{file_path}.yaml")
                try:
                    data = yaml.safe_load(file)
                    # Process YAML data
                    output = {}
                    for item in data:
                        output[item] = {
                            x.lower().strip()
                            for i, x in enumerate(data[item]['Tags'])
                        }
                    self.loaded_tags[key] = output
                except yaml.YAMLError as exc:
                    print(exc)

        # Handle missing files
        if not os.path.isfile(yaml_file_path) and not os.path.isfile(
                txt_file_path):
            self.missing_tags.add(file_path)

        # Return final tags
        return self.loaded_tags.get(key) if self.loaded_tags.get(
            key) else []


# Class for selecting tags from <yaml:[tag]> notation
class TagSelector:

    # Initialize class
    def __init__(self, tag_loader, options):
        self.tag_loader = tag_loader
        self.previously_selected_tags = {}
        self.used_values = {}
        self.selected_options = dict(options).get('selected_options', {})
        self.verbose = dict(options).get('verbose', False)
        self.cache_files = dict(options).get('cache_files', True)

    # Select a value from a list of candidates
    def select_value_from_candidates(self, candidates):
        if len(candidates) == 1:
            # Only one value, return it
            if self.verbose:
                print(f'UmiAI: Only one value {candidates} found. Returning it.')
            self.used_values[candidates[0]] = True
            return candidates[0]

        if len(candidates) > 1:
            # Check for unused value
            for candidate in candidates:
                if candidate not in self.used_values:
                    self.used_values[candidate] = True
                    return candidate

            # No unused, return random
            random.shuffle(candidates)
            if self.verbose:
                print(f'UmiAI: All values in {candidates} were used. Returning random tag ({candidates[0]}).')
            return candidates[0]

    # Get tag choice for a tag
    def get_tag_choice(self, parsed_tag, tags):

        # Check if tag has predefined selection
        if self.selected_options.get(parsed_tag.lower()) is not None:
            return tags[self.selected_options.get(parsed_tag.lower())]

        # Handle no tags
        if len(tags) == 0:
            return ""

        # Shuffle tags and select one
        shuffled_tags = list(tags)
        random.shuffle(shuffled_tags)
        return self.select_value_from_candidates(shuffled_tags)

    # Get tag group choice
    def get_tag_group_choice(self, parsed_tag, groups, tags):

        # Get negative tag groups (start with --)
        neg_groups = [x.strip().lower() for x in groups if x.startswith('--')]
        neg_groups_set = {x.replace('--', '') for x in neg_groups}

        # Get OR tag groups (contain |)
        any_groups = [{y.strip() for i, y in enumerate(x.lower().split('|'))}
                      for x in groups if '|' in x]

        # Get positive tag groups
        pos_groups = [x.strip().lower() for x in groups if not x.startswith('--') and '|' not in x]
        pos_groups_set = {x for x in pos_groups}

        candidates = []
        for tag in tags:

            # Get tags for this tag
            tag_set = tags[tag]

            # Check positive group condition
            if len(list(pos_groups_set & tag_set)) != len(pos_groups_set):
                continue

            # Check negative group condition
            if len(list(neg_groups_set & tag_set)) > 0:
                continue

            # Check any group condition
            if len(any_groups) > 0:
                any_groups_found = 0
                for any_group in any_groups:
                    if len(list(any_group & tag_set)) == 0:
                        break
                    any_groups_found += 1
                if len(any_groups) != any_groups_found:
                    continue

            # Tag matches conditions, add to candidates
            candidates.append(tag)

        # Select candidate randomly
        if len(candidates) > 0:
            # Print debug
            if self.verbose:
                print(
                    f'UmiAI: Found {len(candidates)} candidates for "{parsed_tag}" with tags: {groups}, '
                    f'first 10: {candidates[:10]}'
                )
            random.shuffle(candidates)
            return self.select_value_from_candidates(candidates)

        # No candidates found
        if self.verbose: print(f'UmiAI: No tag candidates found for: "{parsed_tag}" with tags: {groups}')

        return ""

    # Select
    def select(self, tag, groups=None):

        # Track previously selected tags
        self.previously_selected_tags.setdefault(tag, 0)

        # Validate tag
        if (tag.count(':') == 2) or (len(tag) < 2):
            return False

        # Check if tag has been selected too many times
        if self.previously_selected_tags.get(tag) < 50000:

            # Increment selection count
            self.previously_selected_tags[tag] += 1

            # Parse the tag
            parsed_tag = parse_tag(tag)

            # Load tags for this tag
            tags = self.tag_loader.load_tags(parsed_tag, self.verbose, self.cache_files)

            # Handle tag groups
            if groups and len(groups) > 0:
                return self.get_tag_group_choice(parsed_tag, groups, tags)

            # Select tag if tags found
            if len(tags) > 0:
                return self.get_tag_choice(parsed_tag, tags)

            # No tags found
            else:
                print(f'UmiAI: No tags found in wildcard file "{parsed_tag}" or file does not exist')

        # Handle too many selections
        if self.previously_selected_tags.get(tag) == 50000:
            # Increment count
            self.previously_selected_tags[tag] += 1

            # Print warning
            print(
                f'Processed more than 50000 hits on "{tag}". '
                f'This probaly is a reference loop. Inspect your tags and remove any loops.')

        # Invalid tag
        return False


# Class for replacing tags in text
class TagReplacer:

    # Initialize
    def __init__(self, tag_selector, options):
        self.tag_selector = tag_selector
        self.options = options
        self.wildcard_regex = re.compile('((__|<)(.*?)(__|>))')
        self.opts_regexp = re.compile('(?<=\[)(.*?)(?=\])')

    # Replace a single wildcard match
    def replace_wildcard(self, matches):

        # Validate match
        if matches is None or len(matches.groups()) == 0:
            return ""

        # Get tag
        match = matches.groups()[2]
        match_and_opts = match.split(':')

        # Handle tag options
        if (len(match_and_opts) == 2):
            selected_tags = self.tag_selector.select(match_and_opts[0], self.opts_regexp.findall(match_and_opts[1]))
        else:
            global_opts = self.opts_regexp.findall(match)
            if len(global_opts) > 0:
                selected_tags = self.tag_selector.select(ALL_KEY, global_opts)
            else:
                selected_tags = self.tag_selector.select(match)

        # Replace tag
        if selected_tags:
            return selected_tags
        return matches[0]

    # Replace wildcards recursively
    def replace_wildcard_recursive(self, prompt):
        p = self.wildcard_regex.sub(self.replace_wildcard, prompt)
        while p != prompt:
            prompt = p
            p = self.wildcard_regex.sub(self.replace_wildcard, prompt)

        return p

    # Main replace method
    def replace(self, prompt):
        return self.replace_wildcard_recursive(prompt)


# Class for replacing {1$$this | that} dynamic notation
class DynamicPromptReplacer:

    # Compile regex to find {} combinations
    def __init__(self):
        self.re_combinations = re.compile(r"\{([^{}]*)\}")

    # Get weight for a variant
    def get_variant_weight(self, variant):
        split_variant = variant.split("%")
        if len(split_variant) == 2:
            num = split_variant[0]
            try:
                return int(num)
            except ValueError:
                print(f'{num} is not a number')
        return 0

    # Get cleaned variant text
    def get_variant(self, variant):
        split_variant = variant.split("%")
        if len(split_variant) == 2:
            return split_variant[1]
        return variant

    # Parse quantity range
    def parse_range(self, range_str, num_variants):
        # Parse range string
        if range_str is None:
            return None

        parts = range_str.split("-")
        if len(parts) == 1:
            low = high = min(int(parts[0]), num_variants)
        elif len(parts) == 2:
            low = int(parts[0]) if parts[0] else 0
            high = min(int(parts[1]),
                       num_variants) if parts[1] else num_variants
        else:
            raise Exception(f"Unexpected range {range_str}")

        return min(low, high), max(low, high)

        # Replace a {} combination

    def replace_combinations(self, match):

        # Validate match
        if match is None or len(match.groups()) == 0:
            return ""

        # Get variants
        combinations_str = match.groups()[0]
        variants = [s.strip() for s in combinations_str.split("|")]

        # Get weights and clean variants
        weights = [self.get_variant_weight(var) for var in variants]
        variants = [self.get_variant(var) for var in variants]

        # Parse quantity
        splits = variants[0].split("$$")
        quantity = splits.pop(0) if len(splits) > 1 else str(1)
        variants[0] = splits[0]

        # Get quantity range
        low_range, high_range = self.parse_range(quantity, len(variants))

        # Select random quantity
        quantity = random.randint(low_range, high_range)

        # Calculate weights
        summed = sum(weights)
        zero_weights = weights.count(0)
        weights = list(
            map(lambda x: (100 - summed) / zero_weights
            if x == 0 else x, weights))

        # Randomly pick variants
        try:
            # print(f"choosing {quantity} tag from:\n{' , '.join(variants)}")
            picked = []
            for x in range(quantity):
                choice = random.choices(variants, weights)[0]
                picked.append(choice)

                # Remove picked variant
                index = variants.index(choice)
                variants.pop(index)
                weights.pop(index)

            return " , ".join(picked)
        except ValueError as e:
            return ""

    # Main replace method
    def replace(self, template):
        if template is None:
            return None

        return self.re_combinations.sub(self.replace_combinations, template)


# Class for generating options
class OptionGenerator:

    # Initialize
    def __init__(self, tag_loader):
        self.tag_loader = tag_loader

    # Get configurable option tags
    def get_configurable_options(self):
        return self.tag_loader.load_tags('configuration')

    # Get choices for a tag
    def get_option_choices(self, tag):
        return self.tag_loader.load_tags(parse_tag(tag))

    # Parse user options
    def parse_options(self, options):

        # Dictionary to store presets
        tag_presets = {}

        # Loop through configurable tags
        for i, tag in enumerate(self.get_configurable_options()):

            # Parse tag
            parsed_tag = parse_tag(tag)

            # Get index of option
            location = get_index(self.tag_loader.load_tags(parsed_tag), options[i])

            # Add preset if found
            if location is not None:
                tag_presets[parsed_tag.lower()] = location

        return tag_presets


# Class for generating prompts
class PromptGenerator:

    # Initialize
    def __init__(self, options):
        self.tag_loader = TagLoader()
        self.tag_selector = TagSelector(self.tag_loader, options)
        self.replacers = [
            TagReplacer(self.tag_selector, options),
            DynamicPromptReplacer()
        ]
        self.verbose = dict(options).get('verbose', False)

    # Apply all replacers to prompt
    def use_replacers(self, prompt):
        for replacer in self.replacers:
            prompt = replacer.replace(prompt)
        return prompt

    # Generate single prompt
    def generate_single_prompt(self, original_prompt):

        # Recursively apply replacers
        previous_prompt = original_prompt
        start = time.time()
        prompt = self.use_replacers(original_prompt)
        while previous_prompt != prompt:
            previous_prompt = prompt
            prompt = self.use_replacers(prompt)

        # Print time
        end = time.time()
        if self.verbose:
            print(f"Prompt generated in {end - start} seconds")

        return prompt


def generate_prompts(prompt, verbose=False, cache_files=True):
    # Original prompt and negatives
    original_prompt = prompt

    # Parse options
    options = {
        'verbose': verbose,
        'cache_files': cache_files,
    }
    # Create generators
    prompt_generator = PromptGenerator(options)

    # Generate prompt
    prompt = prompt_generator.generate_single_prompt(original_prompt)

    # Store original prompt
    # p.extra_generation_params["Original prompt"] = original_prompt

    return prompt


def find_our_negs(ogprompt):
    # This regex will match text between pairs of '**'
    fxprompt = r'\*\*([^\*]+)\*\*'
    # Replace the matched text with an empty string
    neg = ", ".join(re.findall(fxprompt, ogprompt))
    cleaned_text = re.sub(fxprompt, '', ogprompt)
    return cleaned_text, neg


class UmiTextGen:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "placeholder": " \n "}),
                "separator": ("STRING", {"multiline": False, "default": ", "}),
                "cache_files": (["on", "off"],),
                "verbose": (["off", "on"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "append_to_this_pos": ("STRING", {"forceInput": True}),
                "append_to_this_neg": ("STRING", {"forceInput": True}),
                }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("pos", "neg")
    FUNCTION = "encode"

    CATEGORY = "UmiAI"

    def __init__(self):
        pass

    def encode(self, text, cache_files, verbose, seed, append_to_this_pos="", append_to_this_neg="", separator=", "):
        if not append_to_this_pos:
            append_to_this_pos = ""
        if not append_to_this_neg:
            append_to_this_neg = ""

        cache_files = True if cache_files == "on" else False
        verbose = True if verbose == "on" else False
        prompt = generate_prompts(text, cache_files=cache_files, verbose=verbose)
        old_prompt = ""
        while prompt != old_prompt:
            old_prompt = prompt
            # Getting rid of useless things that sometimes pops up in my prompts
            prompt = prompt.replace("  ", " ").replace(",,", ",").replace(", ,", ",").replace("\n ", "\n").replace("\n\n\n", "\n\n").replace(" ,", ",").replace("\n, ", "\n").replace("\n ", "\n")

        prompt, negs = find_our_negs(prompt)
        if append_to_this_pos != "":
            prompt = f"{append_to_this_pos}{separator}{prompt}"
        if append_to_this_neg != "":
            negs = f"{append_to_this_neg}{separator}{negs}"

        return (prompt, negs)


def random_line(text):
    lines = text.split("\n")  # split the text into a list of lines
    lines = list(filter(bool, lines))  # remove any empty strings from the list
    line = random.choice(lines)  # randomly choose one line from the list
    return line  # return the chosen line


class RandomLineSelection:
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "placeholder": " \n "}),
                "separator": ("STRING", {"multiline": False, "default": ", "}),
                "cache_files": (["on", "off"],),
                "verbose": (["off", "on"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "append_to_this_pos": ("STRING", {"forceInput": True}),
                "append_to_this_neg": ("STRING", {"forceInput": True}),
                }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("pos", "neg")
    FUNCTION = "encode"

    CATEGORY = "UmiAI"

    def __init__(self):
        pass

    def encode(self, text, cache_files, verbose, seed, append_to_this_pos="", append_to_this_neg="", separator=", "):
        # "seed" exists to make comfy execute this node in every new job and not just cache & reuse the previous output
        r_line = random_line(text)
        cache_files = True if cache_files == "on" else False
        verbose = True if verbose == "on" else False
        line = generate_prompts(r_line, cache_files=cache_files, verbose=verbose)
        old_line = ""
        while line != old_line:
            old_line = line
            line = line.replace("  ", " ").replace(",,", ",").replace(", ,", ",").replace("\n ", "\n").replace("\n\n\n", "\n\n").replace(" ,", ",").replace("\n, ", "\n").replace("\n,", "\n").replace("\n ", "\n").replace(", \n", ", ").replace(",\n", ", ")

        line, negs = find_our_negs(line)
        if append_to_this_pos != "":
            line = f"{append_to_this_pos}{separator}{line}"
        if append_to_this_neg != "":
            negs = f"{append_to_this_neg}{separator}{negs}"

        return (line, negs)


NODE_CLASS_MAPPINGS = {
    "UmiAI - Advanced Prompt Generation - v2": UmiTextGen,
    "UmiAI - Random Line Selection - v2": RandomLineSelection
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UmiAI - Advanced Prompt Generation - v2": "Advanced Prompt Generation - UmiAI - v2",
    "UmiAI - Random Line Selection - v2": "Random Line Selection - UmiAI - v2"
}

