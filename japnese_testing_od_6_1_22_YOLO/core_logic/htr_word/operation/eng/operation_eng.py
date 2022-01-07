from core_logic.htr_word.recognition.misc.eng.htr_english import english_recog
def process_english(img_list):
	try:
		results = []
		confidence_scores = []
		for img in img_list:

			result,confidence_score = english_recog(img)
			results.append(result)
			confidence_scores.append(confidence_score)

		result = ''.join(results)

		return result, confidence_scores
	except Exception as e:
		raise e
