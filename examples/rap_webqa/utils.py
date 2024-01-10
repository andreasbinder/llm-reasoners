import re
from typing import Optional, Union

import io

# TODO suggested improvements
# Python
# import json
# from pathlib import Path
# from tqdm import tqdm
# from itertools import islice

# def load_webqa_dataset(path_to_webqa, split, resume):
#     with open(Path(path_to_webqa), 'r') as file:
#         data = json.load(file)

#     if isinstance(resume, list):
#         iterator = islice(data.items(), resume[0], resume[1])
#     elif isinstance(resume, str):
#         iterator = {resume : data[resume]}.items()
#     elif isinstance(resume, int):
#         iterator = islice(data.items(), resume, resume + 1)

#     selected_data = {
#         key: {
#             'Q': value['Q'],
#             'Guid': value['Guid'],
#             'A': value['A'],
#             'split': value['split'],
#             'Qcate': value.get('Qcate', None),
#             'Keywords_A': 'TBD',
#             'answer': 'TBD',
#             'sources': [],
#             'path': path_to_webqa
#         } for key, value in tqdm(iterator, desc="Loading data")
#     }

#     return selected_data

def load_webqa_dataset(path_to_webqa, split, resume):
    import json
    from pathlib import Path
    from tqdm import tqdm

    with open(Path(path_to_webqa), 'r') as file:
        data = json.load(file)

    from itertools import islice

    if isinstance(resume, list):
        iterator = islice(data.items(), resume[0], resume[1])
    elif isinstance(resume, str):
        iterator = {resume : data[resume]}.items()
    elif isinstance(resume, int):
        iterator = islice(data.items(), resume, resume + 1)

    selected_data = {}
    for key, value in tqdm(iterator, desc="Loading data"):
        new_value = {}
        new_value['Q'] = value['Q']
        new_value['Guid'] = value['Guid']
        new_value['A'] = value['A']
        new_value['split'] = value['split']
        new_value['Qcate'] = value.get('Qcate', None)
        new_value['Keywords_A'] = value.get('Keywords_A', 'TBD')
        # print(key, value['Q'])
        # value.setdefault('Qcate', None)
        # new ones
        #new_value.setdefault('Keywords_A', 'TBD')
        new_value.setdefault('answer', 'TBD')
        new_value.setdefault('sources', [])
        new_value.setdefault('path', path_to_webqa)
        selected_data[key] = new_value

    return selected_data


def find_first_appearance(text, keys):
    keywords = list(keys)
    
    for keyword in keywords:
        if keyword in text:
            return keyword
    
    return None 
   

# def action_selection_prompt(prompt, example, state):
#     with io.StringIO() as f:
#         f.write(prompt["action_selection"]["description"] + "\n") 

#         # give overall question
#         f.write(prompt["general"]["prefix_main"] + example + "\n") 

#         # write action descriptions
#         f.write(prompt["action_selection"]["options"] + "\n") 
#         for idx, a in enumerate(prompt["actions"]):
#             f.write(a + ": " + prompt["actions"][a]["description"] + "\n")

#         # write history
#         # only write if history exists
#         if state != []:
#             f.write(prompt["action_selection"]["history"] + "\n") 
#             for idx, a in enumerate(prompt["actions"]):
#                 # do not print action with no history
#                 if any(s.state_type == a for s in state):
#                     # f.write(a + ": " + "\n")
#                     # for idx, s in enumerate(state):
#                     #     if a == s.state_type:
#                     #         #g.write(s[0] + " "+ s[1] + "\n")
#                     #         f.write(get_history(s, a) + "\n")
#                     f.write(get_history(state, a)) # TODO 

#         # output format
#         f.write(prompt["action_selection"]["output_format"] + "\n") 
#         model_input = f.getvalue()
#     return model_input

def format_actions(config, actions):
    out = ""
    for idx, a in enumerate(actions):
        out += a + ": " + config["actions"][a]["description"] + "\n"
    return out


def format_context(state):

    #snippets = [snippet for state in state_list for snippet in state.retrieved_snippets]
    state_functions = {
        'RETRIEVE': lambda state: "\n".join(f"- {sentence}" for sentence in state.retrieved_snippets),
        'HYPOTHESIS': lambda state: "- Proposition: " + state.proposition + " -> " + "Comment: " + state.comment[0],
        # ... other states
    }


    out = ""
    for idx, s in enumerate(state):
        out += f"{state_functions[s.state_type](s) }" + "\n"
        #out += f"{idx}) {c}" + "\n"
    return out

def action_selection_prompt(config, question, state, available_actions):

    prompts = config["action_selection"]["prompts"]
    #available_actions = config["action_selection"]["available_actions"]

    if state == []:
        prompt = prompts["base"]
        prompt = prompt.format(
            overall_question=question,
            available_actions=format_actions(config, available_actions)
        )
    else:
        prompt = prompts["subsequent"]
        prompt = prompt.format(
            overall_question=question,
            available_actions=format_actions(config, available_actions),
            context=format_context(state)
        )
        
    return prompt

# def action_prompt(prompt, example, state, action):
    
#     if action == "HYPOTHESIS":
#         action = "ANSWER"

#     with io.StringIO() as g:
#         g.write(prompt["actions"][action]["description"] + "\n") 

#         # give overall question
#         g.write(prompt["general"]["prefix_main"] + example + "\n") 

#         # write history
#         # only write if history exists
#         if state != []:
#             g.write(prompt["actions"][action]["history"] + "\n") 
#             for idx, a in enumerate(prompt["actions"]):
#                 # do not print action with no history
#                 if any(s.state_type == a for s in state):
#                     #g.write(a + ": " + "\n")
#                     # TODO long term solution should be cleaner
#                     # for idx, s in enumerate(state):
#                     #     if a == s.state_type:
#                     #         #g.write(s[0] + " "+ s[1] + "\n")
#                     #         g.write(get_history(s, a) + "\n")
#                     # TODO right now anyway only retrieve has history
#                     g.write(get_history(state, a)) # TODO 

#         # # write examples
#         # g.write(prompt["actions"][action]["examples"]["prefix"] + "\n") 
#         # for idx, (parent_question, child_question) in enumerate(prompt["actions"][action]["examples"]["data"]):

#         #     g.write("Parent Question: " + prompt["actions"][action]["examples"]["data"][idx]["parent_question"] + "\n")
#         #     g.write("Child Question: " + prompt["actions"][action]["examples"]["data"][idx]["child_question"] + "\n")

#         #g.write("Please write a query that uses the context to get new information.")

#         # output format
#         g.write(prompt["actions"][action]["output_format"] + "\n") 
#         # g.write("Parent Question: " + example + "\n")
#         # g.write("Child Question: ")
#         model_input = g.getvalue()
#     return model_input      

def action_prompt(config, question, state, action):
    
    prompts = config["actions"][action]["prompts"]
    

    if state == []:
        prompt = prompts["base"]
        prompt = prompt.format(
            overall_question=question
        )
    else:
        prompt = prompts["subsequent"]
        prompt = prompt.format(
            overall_question=question,
            context=format_context(state)
        )
        
    return prompt

# def evaluation_prompt(action_prompt, prompt, example, state, action):
    
#     # old code
#     # with io.StringIO() as f:
#     #         f.write(self.useful_prompt["input"])
#     #         f.write(self.useful_prompt["question_prefix"] + self.example + "\n")
#     #         for idx, (q, _, _) in enumerate(state):
#     #             f.write(self.useful_prompt["subquestion_prefix"].format(idx + 1) + " " + q + "\n")
#     #         f.write(self.useful_prompt["new_subquestion_prefix"].format(len(state) + 1) + " " + action + "\n")
#     #         f.write(self.useful_prompt["useful_prefix"])
#     #         model_input = f.getvalue()

#     with io.StringIO() as g:
#         g.write(prompt["general"]["description"] + "\n") 

#         # give overall question
#         g.write(prompt["general"]["prefix_main"] + example + "\n") 

#         # write history
#         # only write if history exists
#         if state != []:
#             g.write(prompt["general"]["history"] + "\n") 
#             for idx, a in enumerate(action_prompt["actions"]):
#                 # do not print action with no history
#                 if any(s.state_type == a for s in state):
#                     #g.write(a + ": " + "\n")
#                     # for idx, s in enumerate(state):
#                     #     if a == s.state_type:
#                     #         #g.write(s[0] + " "+ s[1] + "\n")
#                     #         g.write(get_history(s, a) + "\n")
#                     g.write(get_history(state, a)) # TODO 

#         # # write examples
#         # g.write(prompt["actions"][action]["examples"]["prefix"] + "\n") 
#         # for idx, (parent_question, child_question) in enumerate(prompt["actions"][action]["examples"]["data"]):

#         #     g.write("Parent Question: " + prompt["actions"][action]["examples"]["data"][idx]["parent_question"] + "\n")
#         #     g.write("Child Question: " + prompt["actions"][action]["examples"]["data"][idx]["child_question"] + "\n")

#         # output format
#         g.write(prompt["general"]["prefix_action"] + " " + action + "\n") 
#         # suffix_action
#         g.write(prompt["general"]["suffix_action"] + "\n")  
        
#         model_input = g.getvalue()
#     return model_input

def evaluation_prompt(config, prompt, question, state, action):
    
    keyword, details = action

    prompts = config["evaluation"]["prompts"]
    #available_actions = config["action_selection"]["available_actions"]

    chosen_action = config["actions"][keyword]["prompts"]["evaluation"]

    if keyword == "RETRIEVE":
        chosen_action = chosen_action.format(
            query=details
        )
    if keyword == "HYPOTHESIS":
        chosen_action = chosen_action.format(
            hypothesis=details
        )

    if state == []:
        prompt = prompts["base"]
        prompt = prompt.format(
            overall_question=question,
            chosen_action=chosen_action
        )
    else:
        prompt = prompts["subsequent"]
        prompt = prompt.format(
            overall_question=question,
            chosen_action=chosen_action,
            context=format_context(state)
        )
        
    return prompt

def answer_prompt(prompt, example, state, action):

    with io.StringIO() as g:
        g.write(prompt["actions"][action]["description"] + "\n") 

        # give overall question
        g.write(prompt["general"]["prefix_main"] + example + "\n") 

        # write history
        # only write if history exists
        if state != []:
            g.write(prompt["actions"][action]["history"] + "\n") 
            for idx, a in enumerate(prompt["actions"]):
                # do not print action with no history
                if any(s.state_type == a for s in state):
                    #g.write(a + ": " + "\n")
                    # for idx, s in enumerate(state):
                    #     if a == s.state_type:
                    #         #g.write(s[0] + " "+ s[1] + "\n")
                    #         g.write(get_history(s, a) + "\n")
                    g.write(get_history(state, a)) # TODO 

        # output format
        g.write(prompt["actions"][action]["output_format"] + "\n") 
        # g.write("Parent Question: " + example + "\n")
        # g.write("Child Question: ")
        model_input = g.getvalue()
    return model_input      

def state_transition_prompt(config, question, state, action, details):
    prompts = config["actions"][action]["prompts"]["state_transition"]

    if state == []:
        
        prompt = prompts["base"]
        prompt = prompt.format(
            overall_question=question,
            details=details
        )
    else:
        prompt = prompts["subsequent"]
        prompt = prompt.format(
            overall_question=question,
            context=format_context(state),
            details=details
        )
        
    return prompt

def hypothesis_prompt(config, question, state, action, details):
    prompts = config["actions"][action]["prompts"]["state_transition"]

    # prompt = "You are presented with a proposed conclusion to an overall question and a reasoning path to it.\nThe overall question is: {overall_question}\nThe proposed conclusion is: {details}\nIs the proposed conclusion correct?\nExplain your decision."
    # prompt = "You are presented with a proposed conclusion to an overall question and a reasoning path to it.\nThe overall question is: {overall_question}\nThis is the available context:\n{context}\nThe proposed conclusion is: {details}\nIs the proposed conclusion correct?\nExplain your decision."

    if state == []:
        
        #prompt = "You are presented with a proposed conclusion to an overall question. Is the proposed conclusion correct? Output 'Yes' or 'No', and a reason.\nThe overall question is: {overall_question}\nThe proposed conclusion is: {details}\nYour comment is: "
        prompt = prompt.format(
            overall_question=question,
            details=details
        )
    else:
        #prompt = "You are presented with a proposed conclusion to an overall question. Is the proposed conclusion correct? Output 'Yes' or 'No', and a reason.\nThe overall question is: {overall_question}\nThis is the available context:\n{context}\nThe proposed conclusion is: {details}\nYour comment is: "
        prompt = prompt.format(
            overall_question=question,
            context=format_context(state),
            details=details
        )
        
    return prompt

# def hypothesis_prompt(prompt, example, state, action, details):

#     action = "HYPOTHESIS"
#     with io.StringIO() as g:
#         g.write(prompt["actions"][action]["description"] + "\n") 

#         # give overall question
#         g.write(prompt["general"]["prefix_main"] + example + "\n") 

#         # write history
#         # only write if history exists
#         if state != []:
#             g.write(prompt["actions"][action]["history"] + "\n") 
#             for idx, a in enumerate(prompt["actions"]):
#                 # do not print action with no history
#                 if any(s.state_type == a for s in state):
#                     #g.write(a + ": " + "\n")
#                     # for idx, s in enumerate(state):
#                     #     if a == s.state_type:
#                     #         #g.write(s[0] + " "+ s[1] + "\n")
#                     #         g.write(get_history(s, a) + "\n")
#                     g.write(get_history(state, a)) # TODO 

#         # suggested answer
#         g.write(prompt["actions"][action]["suggestion"] + details + "\n") 

#         # output format
#         g.write(prompt["actions"][action]["output_format"] + "\n") 
#         # g.write("Parent Question: " + example + "\n")
#         # g.write("Child Question: ")
#         model_input = g.getvalue()
#     return model_input      


def get_history(state_list, action):
    # if action == "RETRIEVE":
    #     return state.retrieved_snippets

    if action == "RETRIEVE":
        snippets = [snippet for state in state_list for snippet in state.retrieved_snippets]
        out = ""
        for idx, snippet in enumerate(snippets):
            out += f"{idx}) {snippet}" + "\n"
        return out
    if action == "HYPOTHESIS":
        out = ""
        out += "\n"
        out += "Here are some suggested answers, with some comments on it following the arrow symbol \"->\"" + "\n"
        for state in state_list:
            for proposition, comment in zip(state.proposition, state.comment):
                out += f"{proposition} -> {comment}" + "\n"
        # for idx, snippet in enumerate(snippets):
        #     out += f"{idx}) {snippet}" + "\n"
        return out

    return ""
    


def retrieve_answer(output: Union[list, str]) -> Optional[str]:
    '''
    output should be a world_model.GSM8kState if being a list
    '''
    if isinstance(output, list):
        output = output[-1].sub_answer
    match = re.match(r'.*The answer is .*?([ $.0-9,\-]+).*\..*', output)
    if match is None:
        return None
    answer = match[1].replace(',', '').replace('$', '').replace(' ', '')
    if '=' in answer:
        answer = answer[answer.rindex('=') + 1:]
    return answer


def retrieve_answer_from_dataset(answer: str) -> str:
    return re.match(r'[\S\s]*#### (.*)$', answer)[1]


def judge_answer(output: Optional[str], answer: str) -> bool:
    if output is None:
        return False
    try:
        output = int(output)
        answer = int(answer)
        return output == answer
    except ValueError:
        pass
    try:
        output = float(output)
        answer = float(answer)
        return output == answer
    except ValueError:
        pass
    return output == answer
