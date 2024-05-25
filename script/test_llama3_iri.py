import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import time

from groq import Groq

# Setting hyperparameters
n_sample = 500
temperature = 0.7
model = 'llama3'
date = '0508'
test = True
# Reversing the score of inverse-scored items
reverse = True
# the path to preserve the output
empathy_result_path = './result/{}_{}.xlsx'.format(model, date)


client = Groq(
    api_key="YOUR_API_KEY",
)

model_dict = {"llama3": "llama3-8b-8192"}

if not os.path.exists(empathy_result_path):
    df = pd.DataFrame(columns=['26', '5', '7', '16', '1', '12', '23', '28', '15', '11', '21', '3', '8', '25', '9', '18', '2', '22', '4', '14', '20', '27', '10', '6', '19', '17', '13', '24'])
    df.to_excel(empathy_result_path, index=False)



question_prompt = '''1. When I am reading an interesting story or novel, I imagine how I would feel if the events in the story were happening to me.
            2. I really get involved with the feelings of the characters in a novel.
            3. I am usually objective when I watch a movie or play, and I don\'t often get completely caught up in it.
            4. After seeing a play or movie, I have felt as though I were one of the characters.
            5. I daydream and fantasize, with some regularity, about things that might happen to me.
            6. Becoming extremely involved in a good book or movie is somewhat rare for me.
            7. When I watch a good movie, I can very easily put myself in the place of a leading character.
            8. Before criticizing somebody, I try to imagine how I would feel if I were in their place.
            9. If I'm sure I'm right about something, I don't waste much time listening to other people's arguments. 
            10. I sometimes try to understand my friends better by imagining how things look from their perspective.
            11. I believe that there are two sides to every question and try to look at them both.
            12. I sometimes find it difficult to see things from the "other guy's" point of view. 
            13. I try to look at everybody's side of a disagreement before I make a decision.
            14. When I'm upset at someone, I usually try to "put myself in his shoes" for a while.
            15. When I see someone being taken advantage of, I feel kind of protective toward them.
            16. When I see someone being treated unfairly, I sometimes don't feel very much pity for them.
            17. I often have tender, concerned feelings for people less fortunate than me.
            18. I would describe myself as a pretty soft-hearted person.
            19. Sometimes I don't feel sorry for other people when they are having problems.
            20. Other people's misfortunes do not usually disturb me a great deal.
            21. I am often quite touched by things that I see happen.
            22. When I see someone who badly needs help in an emergency, I go to pieces.
            23. I sometimes feel helpless when I am in the middle of a very emotional situation.
            24. In emergency situations, I feel apprehensive and ill-at-ease.
            25. I am usually pretty effective in dealing with emergencies.
            26. Being in a tense emotional situation scares me.
            27. When I see someone get hurt, I tend to remain calm.
            28. I tend to lose control during emergencies.'''


rule_prompt = """Please answer the quesion by giving a number from 0-4: 
0 indicates "it does not describe me well", 
1 indicates "it describes me poorly", 
2 indicates "it describes me somewhat", 
3 indicates "it describes me moderately", 
4 indicates "it describes me very well".
You should only give the score with no reasons.
The form of the answer should be: score1,score2,score3..."""


openness = ['Openness to experience is a general appreciation for art, emotion, adventure, unusual ideas, imagination, curiosity, and variety of experience.',
            'People who are open to experience are intellectually curious, open to emotion, sensitive to beauty, and willing to try new things. They tend to be, when compared to closed people, more creative and more aware of their feelings. They are also more likely to hold unconventional beliefs. Open people can be perceived as unpredictable or lacking focus, and more likely to engage in risky behaviour or drug-taking. Moreover, individuals with high openness are said to pursue self-actualisation specifically by seeking out intense, euphoric experience.',
            'People with low openness want to be fulfilled by persevering and are characterised as pragmatic and data-driven, and sometimes even perceived to be dogmatic and closed-minded.']

conscientiousness = ['Conscientiousness is a tendency to be self-disciplined, act dutifully, and strive for achievement against measures or outside expectations. It is related to people\'s level of impulse control, regulation, and direction.',
                     'High conscientiousness is often perceived as being stubborn and focused. High conscientiousness indicates a preference for planned rather than spontaneous behaviour.',
                     'Low conscientiousness is associated with flexibility and spontaneity, but can also appear as sloppiness and lack of reliability.']

extraversion = ['Extraversion is characterised by breadth of activities , surgery from external activities or situations, and energy creation from external means.',
                'Extraverts enjoy interacting with people, and are often perceived as energetic. They tend to be enthusiastic and action-oriented. They possess high group visibility, like to talk, and assert themselves. Extraverts may appear more dominant in social settings, as opposed to introverts in that setting.',
                'Introverts have lower social engagement and energy levels than extraverts. They tend to seem quiet, low-key, deliberate, and less involved in the social world. Their lack of social involvement should not be interpreted as shyness or depression; but as greater independence of their social world than extroverts. Introverts need less stimulation, and more time alone than extroverts. This does not mean that they are unfriendly or antisocial; rather, they are aloof and reserved in social situations.']

agreeableness = ['Agreeableness is the general concern for social harmony.',
                 'Agreeable individuals value getting along with others. They are generally considerate, kind, generous, trusting and trustworthy, helpful, and willing to compromise their interests with others. Agreeable people also have an optimistic view of human nature.',
                 'Disagreeable individuals place self-interest above getting along with others. They are generally unconcerned with others\' well-being and are less likely to extend themselves to other people. Sometimes their skepticism about others\' motives causes them to be suspicious, unfriendly, and uncooperative. Disagreeable people are often competitive or challenging, which can be seen as argumentative or untrustworthy.']

neuroticism = ['Neuroticism is the tendency to have strong negative emotions, such as anger, anxiety, or depression. It is sometimes called emotional instability, or is reversed and referred to as emotional stability. Neuroticism is associated with low tolerance for stress or strongly disliked changes.',
               'Neurotic people are emotionally reactive and vulnerable to stress. They are more likely to interpret ordinary situations as threatening. They can perceive minor frustrations as hopelessly difficult. Their negative emotional reactions tend to stay for unusually long periods of time, which means they are often in a bad mood. Neurotic people may display more skin-conductance reactivity than calm and composed people. These problems in emotional regulation can make a neurotic person think less clearly, make worse decisions, and cope less effectively with stress. Being disappointed with one\'s life achievements can make one more neurotic and increase one\'s chances of falling into clinical depression. Neurotic individuals tend to experience more negative life events, but neuroticism also changes in response to positive and negative life experiences. Neurotic people tend to have worse psychological well-being.',
               'Less neurotic individuals are less easily upset and are less emotionally reactive. They tend to be calm, emotionally stable, and free from persistent negative feelings. Freedom from negative feelings does not mean that low scorers experience a lot of positive feelings; that is related to extraversion instead.']


def sample_discrete_normal_distribution(mean=3, std=1.0, size=1):
    """
    Sample from a normal distribution with specified mean and standard deviation,
    then round the samples to the nearest integer within the specified range [1, 5].
    """
    samples = np.random.normal(mean, std, size)
    samples = np.round(samples).clip(1, 5)
    return samples


if test:
    for i in tqdm(range(n_sample)):

        i1 =  int(sample_discrete_normal_distribution(mean=3, std=1.0, size=1)[0])
        i2 =  int(sample_discrete_normal_distribution(mean=3, std=1.0, size=1)[0])
        i3 =  int(sample_discrete_normal_distribution(mean=3, std=1.0, size=1)[0])
        i4 =  int(sample_discrete_normal_distribution(mean=3, std=1.0, size=1)[0])
        i5 =  int(sample_discrete_normal_distribution(mean=3, std=1.0, size=1)[0])

        description = ['very low', 'low', 'neutral', 'high', 'very high']

        i1p =  'You are a person with {} openness.'.format(description[i1-1])
        i2p =  'You are a person with {} conscientiousness.'.format(description[i2-1])
        i3p =  'You are a person with {} extraversion.'.format(description[i3-1])
        i4p =  'You are a person with {} agreeableness.'.format(description[i4-1])
        i5p =  'You are a person with {} neuroticism.'.format(description[i5-1])
        
        personality = openness[0] + openness[1] + openness[2] + i1p \
                + conscientiousness[0] + conscientiousness[1] + conscientiousness[2] + i2p \
                + extraversion[0] + extraversion[1] + extraversion[2] + i3p \
                + agreeableness[0] + agreeableness[1] + agreeableness[2] + i4p \
                + neuroticism[0] + neuroticism[1] + neuroticism[2] + i5p
        
        personality_prompt = """You play a respondent who is participating in a survey.\
        {}
        Please answer the questions as concisely as possible.\
        You should only give the score with no reasons. \
        The form of the answer should be: score1,score2,score3...""".format(personality)

        background = personality_prompt +'\n' + rule_prompt + '\n' + \
        'Here are the questions: [' + question_prompt + ']' + '\n' + \
        'Please carefully obey the rule: ' + rule_prompt

        mes = [{"role": "system", "content": personality_prompt},
                {"role": "user", "content": background}]
        
        while True:
            try:
                completion = client.chat.completions.create(model=model_dict[model], messages=mes, max_tokens=200, temperature=temperature)
                break
            except:
                time.sleep(1)
                
        ans = [_.message.content for _ in completion.choices]
        data = [a.split(',') for a in ans]
        
        try:
            new = pd.DataFrame(data, columns=['26', '5', '7', '16', '1', '12', '23', '28', '15', '11', '21', '3', '8', '25', '9', '18', '2', '22', '4', '14', '20', '27', '10', '6', '19', '17', '13', '24'])
        except:
            continue
        previous = pd.read_excel(empathy_result_path)
        df = pd.concat([previous, new], ignore_index=True)
        df.to_excel(empathy_result_path, index=False)


if reverse:
    previous = pd.read_excel(empathy_result_path)
    negative = ['7', '12', '15', '3', '18', '4', '14', '19', '13']
    df = previous.copy()
    for col in negative:
        if col in df.columns:
            df[col] = 4 - df[col]
    empathy_result_completed_path = './result/{}_{}_completed.xlsx'.format(model, date)
    df.to_excel(empathy_result_completed_path, index=False)

    