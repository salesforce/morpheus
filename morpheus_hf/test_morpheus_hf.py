from morpheus import MorpheusHuggingfaceNLI, MorpheusHuggingfaceQA, MorpheusHuggingfaceSummarization

test_morph_nli = MorpheusHuggingfaceNLI('textattack/bert-base-uncased-MNLI')
premise = 'However, in the off-field (sentimental) tournament, the Falcons and Jets have more appealing story lines.'.lower()
hypothesis = 'The Jets and Falcons have boring stories.'.lower()
label = 'contradiction'

print(test_morph_nli.morph(premise, hypothesis, label))

test_morph_qa = MorpheusHuggingfaceQA('deepset/roberta-base-squad2')
context = "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse (\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries."
q_dict = {"question": "In what country is Normandy located?", "id": "56ddde6b9a695914005b9628", "answers": [{"text": "France", "answer_start": 159}, {"text": "France", "answer_start": 159}, {"text": "France", "answer_start": 159}, {"text": "France", "answer_start": 159}], "is_impossible": False}

print(test_morph_qa.morph(q_dict, context))

test_morph_summ = MorpheusHuggingfaceSummarization('facebook/bart-large-cnn')
source =  "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct."
ref = "The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world."

print(test_morph_summ.morph(source, ref))
