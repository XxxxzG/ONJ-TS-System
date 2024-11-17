# https://xiaosheng.blog/2020/08/13/calculate-bleu-and-rouge

# cumulative BLEU scores
from nltk.translate.bleu_score import sentence_bleu
reference = [['this', 'is', 'small', 'test']]
# candidate = ['this', 'is', 'a', 'test']
candidate = ['this', 'is', 'small', 'test']
print('Cumulative 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
print('Cumulative 3-gram: %f' % sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
print('Cumulative 4-gram: %f' % sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))


# from nltk.translate import meteor_score
# reference = ['this is small test']
# candidate = ['this is a test']
# meteor = meteor_score.meteor_score(reference, candidate)
# print('METEOR score is:', meteor)

from rouge import Rouge

candidate = ['i am a student from xx school']  # 预测摘要, 可以是列表也可以是句子
reference = ['i am a student from school on china'] #真实摘要

rouge = Rouge()
rouge_score = rouge.get_scores(hyps=candidate, refs=reference)
print(rouge_score[0]["rouge-1"])
print(rouge_score[0]["rouge-2"])
print(rouge_score[0]["rouge-l"])