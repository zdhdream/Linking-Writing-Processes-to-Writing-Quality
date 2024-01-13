import gc
import os
import itertools
import pickle
import re
import time
from random import choice, choices
from functools import reduce
from tqdm import tqdm
from itertools import cycle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from functools import reduce
from itertools import cycle
from scipy import stats
from scipy.stats import skew, kurtosis
from sklearn import metrics, model_selection, preprocessing, linear_model, ensemble, decomposition, tree
import lightgbm as lgb
import copy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import random
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from itertools import combinations
from scipy.stats import skew
import copy
import joblib
from sklearn.decomposition import PCA
import warnings
import pickle
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
import math

warnings.filterwarnings('ignore', category=UserWarning)


def preprocessing(df, dataset='test'):
    if dataset == 'train':
        add_value = 66231 - 17831 + 500
        # 校准`a0c24719`的时间
        df.loc[(df['id'] == 'a0c24719') & (df['event_id'] > 68), 'down_time'] += add_value
        df.loc[(df['id'] == 'a0c24719') & (df['event_id'] > 68), 'up_time'] += add_value
    # 针对down_event构建滞后特征
    for i in range(1, 4):
        df[f'down_event_shift{i}'] = df.groupby('id')['down_event'].shift(i)
    df['need_drop'] = np.zeros(len(df))
    # 将down_event == Shift 事件的need_drop标记为1,其他事件标记为0
    df.loc[
        (df['down_event'] == 'Shift') & (df['down_event_shift1'] == 'Shift') & (df['down_event_shift2'] == 'Shift') & (
                df['down_event_shift3'] == 'Shift'), 'need_drop'] = 1
    # 删除need_drop == 1
    df = df[df['need_drop'] == 0]
    # 根据id将数据重新编号
    df['event_id'] = df.groupby('id').cumcount()
    return df.drop(columns=['down_event_shift1', 'down_event_shift2', 'down_event_shift3', 'need_drop'])


##################################################################
#################### Generation Essay ############################
##################################################################
# essay生成：普通版本和带大写版本
class EssayConstructor:

    def processingInputs(self, currTextInput):
        # Where the essay content will be stored
        essayText = ""
        # Produces the essay
        for Input in currTextInput.values:
            # Input[0] = activity
            # Input[1] = cursor_position
            # Input[2] = text_change
            # Input[3] = id
            # If activity = Replace
            if Input[0] == 'Replace':
                # splits text_change at ' => '
                replaceTxt = Input[2].split(' => ')
                # DONT TOUCH
                essayText = essayText[:Input[1] - len(replaceTxt[1])] + replaceTxt[1] + essayText[Input[1] - len(
                    replaceTxt[1]) + len(replaceTxt[0]):]
                continue

            # If activity = Paste
            if Input[0] == 'Paste':
                # DONT TOUCH
                essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
                continue

            # If activity = Remove/Cut
            if Input[0] == 'Remove/Cut':
                # DONT TOUCH
                essayText = essayText[:Input[1]] + essayText[Input[1] + len(Input[2]):]
                continue

            # If activity = Move...
            if "M" in Input[0]:
                # Gets rid of the "Move from to" text
                croppedTxt = Input[0][10:]
                # Splits cropped text by ' To '
                splitTxt = croppedTxt.split(' To ')
                # Splits split text again by ', ' for each item
                valueArr = [item.split(', ') for item in splitTxt]
                # Move from [2, 4] To [5, 7] = (2, 4, 5, 7)
                moveData = (
                    int(valueArr[0][0][1:]), int(valueArr[0][1][:-1]), int(valueArr[1][0][1:]),
                    int(valueArr[1][1][:-1]))
                # Skip if someone manages to activiate this by moving to same place
                if moveData[0] != moveData[2]:
                    # Check if they move text forward in essay (they are different)
                    if moveData[0] < moveData[2]:
                        # DONT TOUCH
                        essayText = essayText[:moveData[0]] + essayText[moveData[1]:moveData[3]] + essayText[
                                                                                                   moveData[0]:moveData[
                                                                                                       1]] + essayText[
                                                                                                             moveData[
                                                                                                                 3]:]
                    else:
                        # DONT TOUCH
                        essayText = essayText[:moveData[2]] + essayText[moveData[0]:moveData[1]] + essayText[
                                                                                                   moveData[2]:moveData[
                                                                                                       0]] + essayText[
                                                                                                             moveData[
                                                                                                                 1]:]
                continue

                # If activity = input
            # DONT TOUCH
            essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
        return essayText

    def getEssays(self, df):
        # Copy required columns
        textInputDf = copy.deepcopy(df[['id', 'activity', 'cursor_position', 'text_change']])
        # Get rid of text inputs that make no change
        textInputDf = textInputDf[textInputDf.activity != 'Nonproduction']
        # construct essay, fast
        tqdm.pandas()
        essay = textInputDf.groupby('id')[['activity', 'cursor_position', 'text_change']].progress_apply(
            lambda x: self.processingInputs(x))
        # to dataframe
        essayFrame = essay.to_frame().reset_index()
        essayFrame.columns = ['id', 'essay']
        # Returns the essay series
        return essayFrame


def getEssays_with_upper(df):
    df['down_event_shift'] = df.groupby('id')['down_event'].shift(1)
    textInputDf = df[['id', 'activity', 'cursor_position', 'text_change', 'down_event', 'down_event_shift']]
    valCountsArr = textInputDf['id'].value_counts(sort=False).values
    lastIndex = 0
    essaySeries = pd.Series()
    for index, valCount in enumerate(tqdm(valCountsArr)):
        capital = False
        currTextInput = textInputDf[
                            ['activity', 'cursor_position', 'text_change', 'down_event', 'down_event_shift']].iloc[
                        lastIndex: lastIndex + valCount]
        lastIndex += valCount
        essayText = ""
        for Input in currTextInput.values:
            if Input[3] == 'CapsLock':
                capital = not capital
            if Input[0] == 'Nonproduction':
                continue
            if Input[0] != 'Nonproduction':
                if (Input[0] == 'Replace') & (Input[4] == 'Shift'):
                    replaceTxt = Input[2].split(' => ')
                    essayText = essayText[:Input[1] - len(replaceTxt[1])] + (replaceTxt[1]).upper() + \
                                essayText[Input[1] - len(replaceTxt[1]) + len(replaceTxt[0]):]
                    continue

                if Input[0] == 'Replace':
                    replaceTxt = Input[2].split(' => ')
                    essayText = essayText[:Input[1] - len(replaceTxt[1])] + replaceTxt[1] + \
                                essayText[Input[1] - len(replaceTxt[1]) + len(replaceTxt[0]):]
                    continue

                if Input[0] == 'Paste':
                    essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
                    continue
                if Input[0] == 'Remove/Cut':
                    essayText = essayText[:Input[1]] + essayText[Input[1] + len(Input[2]):]
                    continue
                if "M" in Input[0]:
                    croppedTxt = Input[0][10:]
                    splitTxt = croppedTxt.split(' To ')
                    valueArr = [item.split(', ') for item in splitTxt]
                    moveData = (int(valueArr[0][0][1:]),
                                int(valueArr[0][1][:-1]),
                                int(valueArr[1][0][1:]),
                                int(valueArr[1][1][:-1]))
                    if moveData[0] != moveData[2]:
                        if moveData[0] < moveData[2]:
                            essayText = essayText[:moveData[0]] + essayText[moveData[1]:moveData[3]] + \
                                        essayText[moveData[0]:moveData[1]] + essayText[moveData[3]:]
                        else:
                            essayText = essayText[:moveData[2]] + essayText[moveData[0]:moveData[1]] + \
                                        essayText[moveData[2]:moveData[0]] + essayText[moveData[1]:]
                    continue
                if capital | ((Input[4] == 'Shift') & (Input[3] == 'q')):
                    essayText = essayText[:Input[1] - len(Input[2])] + Input[2].upper() + essayText[
                                                                                          Input[1] - len(Input[2]):]
                else:
                    essayText = essayText[:Input[1] - len(Input[2])] + Input[2] + essayText[Input[1] - len(Input[2]):]
        essaySeries[index] = essayText
    essaySeries.index = textInputDf['id'].unique()
    return pd.DataFrame(essaySeries, columns=['essay'])


# 两个分位数
def q1(x):
    return x.quantile(0.25)


def q3(x):
    return x.quantile(0.75)


def calculate_entropy(text):
    # 统计每个字符的出现次数
    char_count = {}
    for char in text:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1
    probabilities = [float(char_count[char]) / len(text) for char in char_count]
    entropy = -sum([p * math.log2(p) for p in probabilities])
    return entropy


##################################################################
################# Generation Features ############################
##################################################################
class Preprocessor_v1:

    def __init__(self, seed, essays, essays_with_upper, train_scores=None, tokenizer=None, method='train',
                 save_cols=None):
        self.seed = seed
        self.tokenizer = tokenizer
        self.train_scores = train_scores
        self.save_cols = save_cols
        self.essays = essays
        self.essays_with_upper = essays_with_upper
        self.method = method
        self.activities = ['Input', 'Remove/Cut', 'Nonproduction', 'Replace', 'Paste']
        self.events = ['q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick', 'ArrowLeft', '.', ',',
                       'ArrowDown', 'ArrowUp', 'Enter', 'CapsLock', "'", 'Delete', 'Unidentified']
        self.text_changes = ['q', ' ', 'NoChange', '.', ',', '\n', "'", '"', '-', '?', ';', '=', '/', '\\', ':']
        self.punctuations = ['"', '.', ',', "'", '-', ';', ':', '?', '!', '<', '>', '/', '@', '#', '$', '%', '^', '&',
                             '*', '(', ')', '_', '+']
        self.gaps = [1, 2, 3, 5, 10, 20, 50, 100]
        self.idf = defaultdict(float)
        self.text_changes_dict = {
            'q': 'q',
            ' ': 'space',
            'NoChange': 'NoChange',
            '.': 'full_stop',
            ',': 'comma',
            '\n': 'newline',
            "'": 'single_quote',
            '"': 'double_quote',
            '-': 'dash',
            '?': 'question_mark',
            ';': 'semicolon',
            '=': 'equals',
            '/': 'slash',
            '\\': 'double_backslash',
            ':': 'colon'
        }
        self.AGGREGATIONS = ['nunique', 'count', 'mean', 'std', 'min', 'max', 'first', 'last', 'sem', q1, 'median', q3,
                             'skew', pd.DataFrame.kurt, 'sum']
        self.AGGREGATIONS2 = ['nunique', 'mean', 'std', 'min', 'max', 'first', 'last', q1, 'median', q3, 'sum']
        self.AGGREGATIONS3 = ['nunique', 'mean', 'std', 'min', 'max', 'first', 'last', q1, 'median', q3, 'sum']
        self.AGGREGATIONS4 = ['nunique', 'mean', 'std', 'min', 'max', 'first', 'last', q1, 'median', q3, 'sum']
        self.AGGREGATIONS5 = ['nunique', 'mean', 'std', 'min', 'max', 'first', 'last', q1, 'median', q3, 'sum']

    def activity_counts(self, df):
        """计算每篇论文活动的逆文档频率"""
        tmp_df = df.groupby('id').agg({'activity': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['activity'].values):
            items = list(Counter(li).items())  # 计算每个活动出现的次数 Input: 2 Remove/Cut: 1
            di = dict()
            for k in self.activities:
                di[k] = 0

            di["move_to"] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
                else:
                    di["move_to"] += v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'activity_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols

        cnts = ret.sum(1)  # 计算每行总计数
        # 计算逆文档频率
        for col in cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf

            ret[col] = 1 + np.log(ret[col] / cnts)
            ret[col] *= idf

        return ret

    def event_counts(self, df, colname):
        """计算每篇论文event逆文档频率"""
        tmp_df = df.groupby('id').agg({colname: list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df[colname].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.events:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'{colname}_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols

        cnts = ret.sum(1)

        for col in cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf

            ret[col] = 1 + np.log(ret[col] / cnts)
            ret[col] *= idf

        return ret

    def text_change_counts(self, df):
        """计算每篇论文text_change逆文档频率"""
        tmp_df = df.groupby('id').agg({'text_change': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['text_change'].values):
            items = list(Counter(li).items())
            di = dict()
            for k in self.text_changes:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'text_change_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols

        cnts = ret.sum(1)

        for col in cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf

            ret[col] = 1 + np.log(ret[col] / cnts)
            ret[col] *= idf

        return ret

    def match_punctuations(self, df):
        """计算每篇论文标点符号次数"""
        tmp_df = df.groupby('id').agg({'down_event': list}).reset_index()
        ret = list()
        for li in tqdm(tmp_df['down_event'].values):
            cnt = 0
            items = list(Counter(li).items())
            for item in items:
                k, v = item[0], item[1]
                if k in self.punctuations:
                    cnt += v
            ret.append(cnt)
        ret = pd.DataFrame({'punct_cnt': ret})
        return ret

    def get_input_words(self, df):
        tmp_df = df[(~df['text_change'].str.contains('=>')) & (df['text_change'] != 'NoChange')].reset_index(drop=True)
        tmp_df = tmp_df.groupby('id').agg({'text_change': list}).reset_index()
        tmp_df['text_change'] = tmp_df['text_change'].apply(lambda x: ''.join(x))
        tmp_df['text_change'] = tmp_df['text_change'].apply(lambda x: re.findall(r'q+', x))
        tmp_df['input_word_count'] = tmp_df['text_change'].apply(len)
        tmp_df['input_word_length_mean'] = tmp_df['text_change'].apply(
            lambda x: np.mean([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['input_word_length_max'] = tmp_df['text_change'].apply(
            lambda x: np.max([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['input_word_length_std'] = tmp_df['text_change'].apply(
            lambda x: np.std([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df.drop(['text_change'], axis=1, inplace=True)
        return tmp_df

    def calculate_pauses(self, df, pause_threshold=2000):
        # Compute IKI within each 'id' group
        df['IKI'] = df.groupby('id')['down_time'].diff()  # 计算每篇论文的击键间隔

        # Define pauses
        df['is_pause'] = (df['IKI'] > pause_threshold)  # 击键间隔大于threshold设置为停顿

        # Compute statistics for IKI
        iki_stats = df.groupby('id')['IKI'].agg(['mean', 'median', 'std', 'max']).reset_index().rename(columns={
            'mean': 'iki_mean',
            'median': 'iki_median',
            'std': 'iki_std',
            'max': 'iki_max'
        })

        # Compute pause counts(每篇论文停顿总数)
        pause_counts = df.groupby('id')['is_pause'].sum().reset_index(name='pause_count')

        # Compute average pause time excluding NaNs(每篇论文停顿均值)
        pause_times = df[df['is_pause']].groupby('id')['IKI'].mean().reset_index(name='average_pause_time')

        # Compute total pause time for paragraph(只考虑文本改变时的IKI值,得到文本改变时"IKI"值的总和)
        para_pause_duration = df.groupby('id').apply(
            lambda group: group['IKI'].where(group['text_change'] == '\n').sum()).reset_index(
            name='para_pause_duration')

        # Merge pause features
        pause_features = pause_counts.merge(pause_times, on='id', how='left')
        pause_features = pause_features.merge(para_pause_duration, on='id', how='left')
        pause_features = pause_features.merge(iki_stats, on='id', how='left')

        # Compute total IKI time and exclude NaNs
        total_time = df.groupby('id')['IKI'].sum().reset_index(name='total_time')

        # Merge the total time into pause_features
        pause_features = pause_features.merge(total_time, on='id', how='left')

        # Calculate pause time ratio
        pause_features['pause_time_ratio'] = pause_features['pause_count'] * pause_features['average_pause_time']
        pause_features['pause_time_ratio'] = pause_features['pause_time_ratio'] / pause_features['total_time'].replace(
            0, np.nan)

        # Calculate times between sentences within each 'id' group
        df['sentence_end_IKI'] = df.groupby('id').apply(
            lambda group: group['down_time'].diff().where(group['text_change'].isin(['.', '?', '!']))).reset_index(
            level=0, drop=True)

        # Calculate statistics for times between sentences
        between_sentences_stats = df.groupby('id')['sentence_end_IKI'].agg(['mean', 'std']).reset_index().rename(
            columns={'mean': 'mean_between_sentences_IKI', 'std': 'std_between_sentences_IKI'})

        # Calculate within-word IKI for 'q' characters within each 'id'
        df['within_word_IKI'] = df.groupby('id').apply(
            lambda group: group['down_time'].diff().where(group['text_change'] == 'q')).reset_index(level=0, drop=True)

        # Calculate statistics for within-word IKI
        within_word_stats = df.groupby('id')['within_word_IKI'].agg(['mean', 'std']).reset_index().rename(
            columns={'mean': 'mean_within_word_IKI', 'std': 'std_within_word_IKI'})

        # Calculate between-words IKI for spaces or punctuation followed by 'q'
        df['between_words_IKI'] = df.groupby('id').apply(lambda group: group['down_time'].diff().where(
            group['text_change'].shift().isin([' '] + self.punctuations) & (group['text_change'] == 'q'))).reset_index(
            level=0, drop=True)

        # Calculate statistics for between-words IKI
        between_words_stats = df.groupby('id')['between_words_IKI'].agg(['mean', 'std']).reset_index().rename(
            columns={'mean': 'mean_between_words_IKI', 'std': 'std_between_words_IKI'})

        # Combine all the IKI related features into one DataFrame
        pause_features = pause_features.merge(between_sentences_stats, on='id', how='left')
        pause_features = pause_features.merge(within_word_stats, on='id', how='left')
        pause_features = pause_features.merge(between_words_stats, on='id', how='left')

        return pause_features

    def brute_force_agg(self, df):
        # bruteforce agg
        agg_fe_df = df.groupby("id")[['down_time', 'cursor_position', 'word_count']].agg(
            ['mean', 'std', 'min', 'max', 'last', 'first', 'sem', 'median', 'sum'])
        agg_fe_df.columns = ['_'.join(x) for x in agg_fe_df.columns]
        agg_fe_df = agg_fe_df.add_prefix("tmp_")
        agg_fe_df.reset_index(inplace=True)
        return agg_fe_df

    def duration_features(self, df):
        logs = copy.deepcopy(df)
        logs['up_time_lagged'] = logs.groupby('id')['up_time'].shift(1).fillna(logs['down_time'])  # 释放键滞后特征
        logs['time_diff'] = abs(logs['down_time'] - logs['up_time_lagged']) / 1000

        group = logs.groupby('id')['time_diff']
        initial_pause = logs.groupby('id')['down_time'].first() / 1000
        pauses_half_sec = group.apply(lambda x: ((x > 0.5) & (x < 1)).sum())
        pauses_1_sec = group.apply(lambda x: ((x > 1) & (x < 1.5)).sum())
        pauses_1_half_sec = group.apply(lambda x: ((x > 1.5) & (x < 2)).sum())
        pauses_2_sec = group.apply(lambda x: ((x > 2) & (x < 3)).sum())
        pauses_3_sec = group.apply(lambda x: (x > 3).sum())
        data = pd.DataFrame({
            'id': logs['id'].unique(),
            'initial_pause': initial_pause,
            'pauses_half_sec': pauses_half_sec,
            'pauses_1_sec': pauses_1_sec,
            'pauses_1_half_sec': pauses_1_half_sec,
            'pauses_2_sec': pauses_2_sec,
            'pauses_3_sec': pauses_3_sec,
        }).reset_index(drop=True)
        return data

    def essay_CountVectorizer_and_tfidf(self):
        if self.method == 'train':
            essaysdf = copy.deepcopy(self.essays['essay'])
            essaysdf = pd.DataFrame({'id': essaysdf.index, 'essay': essaysdf.values})
            merged_data = essaysdf.merge(self.train_scores, on='id')
            count_vectorizer = CountVectorizer(ngram_range=(1, 2))
            tokenizer = count_vectorizer.fit_transform(merged_data['essay'])
            y = merged_data['score']
            tokenizer = tokenizer.todense()
            count_vector = pd.DataFrame()
            for i in range(tokenizer.shape[1]):
                L = list(tokenizer[:, i])
                li = [int(x) for x in L]
                count_vector[f'feature {i}'] = li
            df_index = essaysdf['id']
            count_vector.loc[:, 'id'] = df_index

            save_cols = []
            for i in count_vector.columns:
                if sum(count_vector[i] == 0) / len(count_vector) < 0.1:
                    save_cols.append(i)

            return count_vector[save_cols], count_vectorizer, save_cols

        else:
            essaysdf = copy.deepcopy(self.essays['essay'])
            essaysdf = pd.DataFrame({'id': essaysdf.index, 'essay': essaysdf.values})
            tokenizer = self.tokenizer.transform(essaysdf['essay'])
            tokenizer = tokenizer.todense()
            count_vector = pd.DataFrame()
            for i in range(tokenizer.shape[1]):
                L = list(tokenizer[:, i])
                li = [int(x) for x in L]
                count_vector[f'feature {i}'] = li
            df_index = essaysdf['id']
            count_vector.loc[:, 'id'] = df_index
            return count_vector[self.save_cols]

    def other_features(self, df):
        a = pd.DataFrame()
        a['Input_all_ratio'] = df.groupby(['id']).apply(lambda x: sum(x['activity'] != 'Input')) / df.groupby(
            ['id']).apply(lambda x: sum(x['activity'] == 'Input'))
        a['all_q_ratio'] = df.groupby(['id']).apply(lambda x: sum(x['down_event'] != 'q')) / df.groupby(['id']).apply(
            lambda x: sum(x['down_event'] == 'q'))
        activities = ['Input', 'Remove/Cut', 'Nonproduction', 'Replace', 'Paste']
        events_dict = {
            'q': 'q',
            'Space': 'Space',
            'Backspace': 'Backspace',
            'Shift': 'Shift',
            'ArrowRight': 'ArrowRight',
            'Leftclick': 'Leftclick',
            'ArrowLeft': 'ArrowLeft',
            '.': 'fullstop',
            ',': 'comma',
            'ArrowDown': 'ArrowDown',
            'ArrowUp': 'ArrowUp',
            'Enter': 'Enter',
            'CapsLock': 'CapsLock',
            "'": 'single_quote',
            'Delete': 'Delete',
            'Unidentified': 'Unidentified',
        }
        for i in tqdm(activities):
            for j in events_dict:
                a[f'{i}_{events_dict[j]}_count'] = df.groupby('id').apply(
                    lambda x: len(x[(x['activity'] == i) & (x['down_event'] == j)]))
        return a.reset_index()

    def language_error(self, df):
        a = pd.DataFrame()
        df['down_event_shift'] = df.groupby('id')['down_event'].shift(-1)
        letter_upper = df.groupby('id').apply(lambda x: len(
            x[(x['down_event'] == 'CapsLock') | ((x['down_event'] == 'Shift') & (x['down_event_shift'] == 'q'))]))
        a['letter_big_count'] = letter_upper.values
        a['id'] = df['id'].unique()

        essay_df = copy.deepcopy(self.essays)
        essay_df['id'] = essay_df.index

        # 避免将qqq.).切分成多个句子
        # essay_df['essay'] = essay_df['essay'].apply(lambda x:re.sub(r'\.\]|\.\)|\.\}|\?\]|\?\)|\?\}|\!\]|\!\)|\!\}','qq',x))

        essay_df['essay'] = essay_df['essay'].apply(lambda x: re.sub(r'q\.q\.', 'qqq', x))
        essay_df['sent'] = essay_df['essay'].apply(lambda x: re.split('\\.|\\?|\\!', x))
        essay_df = essay_df.explode('sent')  # explode将列表里的元素展开
        essay_df['sent'] = essay_df['sent'].apply(lambda x: x.replace('\n', '').strip())
        essay_df['sent_len'] = essay_df['sent'].apply(lambda x: len(x))
        essay_df = essay_df[essay_df['sent_len'] != 0]
        errors_num = (essay_df.groupby('id').apply(len) - letter_upper).values
        a['error_num'] = errors_num  # 如果句子个数大于大写字母按键次数，那么文章会有语法错误

        return a

    def sentence_error(self):
        essay_df = copy.deepcopy(self.essays)
        essay_df['id'] = essay_df.index
        essay_df['paragraph'] = essay_df['essay'].apply(lambda x: x.split('\n'))
        essay_df = essay_df.explode('paragraph')
        # Number of characters in paragraphs
        essay_df['paragraph_len'] = essay_df['paragraph'].apply(lambda x: len(x))
        essay_df = essay_df[essay_df['paragraph_len'] != 0]
        essay_df['only_space'] = essay_df['paragraph'].apply(lambda x: 'q' not in x)
        essay_df = essay_df[essay_df['only_space'] == False]
        a = pd.DataFrame()
        a['para_error'] = essay_df.groupby('id').apply(
            lambda x: len(x[x['paragraph_len'] < 25]))  # 一个段落字符过少可能不是完整的一句话，可能存在语法错误

        return a.reset_index()

    def language_error_letter(self):
        essay_df = copy.deepcopy(self.essays_with_upper)
        essay_df['id'] = essay_df.index

        # 避免将qqq.).切分成多个句子
        # essay_df['essay'] = essay_df['essay'].apply(lambda x:re.sub(r'\.\]|\.\)|\.\}|\?\]|\?\)|\?\}|\!\]|\!\)|\!\}','qq',x))

        essay_df['essay'] = essay_df['essay'].apply(lambda x: re.sub(r'q\.q\.', 'qqq', x))
        essay_df['sent'] = essay_df['essay'].apply(lambda x: re.split('\\.|\\?|\\!', x))
        essay_df = essay_df.explode('sent')  # explode将列表里的元素展开
        essay_df['sent'] = essay_df['sent'].apply(lambda x: str(x).replace('\n', '').strip())
        essay_df['sent_len'] = essay_df['sent'].apply(lambda x: len(x))
        essay_df = essay_df[essay_df.sent_len != 0].reset_index(drop=True)
        essay_df['language_error_letter'] = essay_df['sent'].apply(lambda x: x[0])
        essay_df['if_q'] = essay_df['language_error_letter'].apply(lambda x: x.lower() == 'q')
        essay_df = essay_df[essay_df['if_q'] == True]
        a = pd.DataFrame()
        a['language_error_letter'] = essay_df.groupby('id').apply(lambda x: len(x[x['language_error_letter'] == 'q']))
        return a.reset_index()

    def R_burst(self, df):
        a = pd.DataFrame()
        df = df[(df['activity'] == 'Input') | (df['activity'] == 'Remove/Cut')].reset_index(drop=True)
        df['activity_shift'] = df.groupby('id')['activity'].shift().fillna(method='bfill')
        df['is_R_burst'] = df['activity'] != df['activity_shift']
        a['revision_count'] = df.groupby('id').apply(lambda x: x['is_R_burst'].sum())
        df['keystroke_duration'] = df.groupby('id')['down_time'].diff()
        df = df[df['keystroke_duration'].notnull()]

        a['revision_count_above2s'] = df.groupby('id').apply(
            lambda x: x[(x['is_R_burst'] == True) & (x['keystroke_duration'] > 2)]['is_R_burst'].sum()).values
        Rburst = df[(df['is_R_burst'] == True) & (df['keystroke_duration'] > 2)]  # &(df['keystroke_duration']>2)
        Rburst_statistic = Rburst.groupby('id').agg({'keystroke_duration': ['mean', 'max', 'sum', 'median']})
        Rburst_statistic.columns = ['_'.join(x) for x in Rburst_statistic.columns]

        return a.merge(Rburst_statistic.reset_index(), on='id', how='left')

    def split_essays_into_sentences(self):
        essay_df = copy.deepcopy(self.essays)
        essay_df['id'] = essay_df.index

        # 避免将qqq.).切分成多个句子
        # essay_df['essay'] = essay_df['essay'].apply(lambda x:re.sub(r'\.\]|\.\)|\.\}|\?\]|\?\)|\?\}|\!\]|\!\)|\!\}','qq',x))
        # 避免将类似于i.e.切分成多个句子
        essay_df['essay'] = essay_df['essay'].apply(lambda x: re.sub(r'q\.q\.', 'qqq', x))
        essay_df['sent'] = essay_df['essay'].apply(lambda x: re.split('\\.|\\?|\\!', x))
        essay_df = essay_df.explode('sent')  # explode将列表里的元素展开
        essay_df['sent'] = essay_df['sent'].apply(lambda x: x.replace('\n', '').strip())  # strip会删除字符串两端的空格
        # Number of characters in sentences
        essay_df['sent_len'] = essay_df['sent'].apply(lambda x: len(x))

        # Number of words in sentences
        essay_df['sent_word_count'] = essay_df['sent'].apply(lambda x: len(x.split(' ')))
        essay_df['sent_word_count_diff'] = essay_df.groupby(['id'])['sent_word_count'].transform(
            lambda x: np.abs(x.diff()))
        essay_df['words_len_above10'] = essay_df['sent'].apply(lambda x: x.split(' '))
        essay_df['words_len_above10'] = essay_df['words_len_above10'].apply(lambda x: sum(len(y) > 10 for y in x))

        essay_df['words_len_5-10'] = essay_df['sent'].apply(lambda x: x.split(' '))
        essay_df['words_len_5-10'] = essay_df['words_len_5-10'].apply(lambda x: sum(5 <= len(y) <= 10 for y in x))

        essay_df['words_len_first'] = essay_df['sent'].apply(lambda x: x.split(' '))
        essay_df['words_len_first'] = essay_df['words_len_first'].apply(lambda x: len(x[0]))

        essay_df = essay_df[essay_df.sent_len != 0].reset_index(drop=True)
        return essay_df

    def compute_sentence_aggregations(self, df):
        sent_agg_df = pd.concat(
            [df[['id', 'sent_len']].groupby(['id']).agg(self.AGGREGATIONS),
             df[['id', 'sent_word_count']].groupby(['id']).agg(self.AGGREGATIONS),
             df[['id', 'sent_word_count_diff']].groupby(['id']).agg(self.AGGREGATIONS2),
             df[['id', 'words_len_above10']].groupby(['id']).agg(self.AGGREGATIONS3),
             df[['id', 'words_len_first']].groupby(['id']).agg(self.AGGREGATIONS4),
             df[['id', 'words_len_5-10']].groupby(['id']).agg(self.AGGREGATIONS5),

             ],
            axis=1)
        sent_agg_df.columns = ['_'.join(x) for x in sent_agg_df.columns]
        sent_agg_df['id'] = sent_agg_df.index
        sent_agg_df = sent_agg_df.reset_index(drop=True)
        sent_agg_df.drop(columns=["sent_word_count_count"], inplace=True)
        sent_agg_df = sent_agg_df.rename(columns={"sent_len_count": "sent_count"})
        return sent_agg_df

    def split_essays_into_paragraphs(self):
        essay_df = copy.deepcopy(self.essays)
        essay_df['id'] = essay_df.index
        essay_df['paragraph'] = essay_df['essay'].apply(lambda x: x.split('\n'))
        essay_df = essay_df.explode('paragraph')
        # Number of characters in paragraphs
        essay_df['paragraph_len'] = essay_df['paragraph'].apply(lambda x: len(x))

        # Number of words in paragraphs
        essay_df['paragraph_word_count'] = essay_df['paragraph'].apply(lambda x: len(x.split(' ')))
        essay_df['paragraph_word_count_diff'] = essay_df.groupby(['id'])['paragraph_word_count'].transform(
            lambda x: np.abs(x.diff()))

        essay_df['para_words_len_above10'] = essay_df['paragraph'].apply(lambda x: x.split(' '))
        essay_df['para_words_len_above10'] = essay_df['para_words_len_above10'].apply(
            lambda x: sum(len(y) > 10 for y in x))

        essay_df['para_words_len_5-10'] = essay_df['paragraph'].apply(lambda x: x.split(' '))
        essay_df['para_words_len_5-10'] = essay_df['para_words_len_5-10'].apply(
            lambda x: sum(5 <= len(y) <= 10 for y in x))

        essay_df['para_words_len_first'] = essay_df['paragraph'].apply(lambda x: x.split(' '))
        essay_df['para_words_len_first'] = essay_df['para_words_len_first'].apply(lambda x: len(x[0]))

        essay_df['num_question'] = essay_df['paragraph'].apply(lambda x: len(re.findall(r'\?', x)))
        essay_df['num_yinyong'] = essay_df['paragraph'].apply(lambda x: len(re.findall(r'\"', x)))

        essay_df = essay_df[essay_df.paragraph_len != 0].reset_index(drop=True)
        # 有些段落可能全部是空格，类似于：'    '
        # essay_df['only_space'] = essay_df['paragraph'].apply(lambda x:'q' not in x)
        # essay_df = essay_df[essay_df['only_space']==False]
        return essay_df

    def compute_paragraph_aggregations(self, df):
        paragraph_agg_df = pd.concat(
            [df[['id', 'paragraph_len']].groupby(['id']).agg(self.AGGREGATIONS), \
             df[['id', 'paragraph_word_count']].groupby(['id']).agg(self.AGGREGATIONS),
             df[['id', 'paragraph_word_count_diff']].groupby(['id']).agg(self.AGGREGATIONS2),
             df[['id', 'para_words_len_above10']].groupby(['id']).agg(self.AGGREGATIONS3),
             df[['id', 'para_words_len_first']].groupby(['id']).agg(self.AGGREGATIONS4),
             df[['id', 'para_words_len_5-10']].groupby(['id']).agg(self.AGGREGATIONS5),
             df[['id', 'num_question']].groupby(['id']).agg(self.AGGREGATIONS5),
             df[['id', 'num_yinyong']].groupby(['id']).agg(self.AGGREGATIONS5),

             ], axis=1)
        paragraph_agg_df.columns = ['_'.join(x) for x in paragraph_agg_df.columns]
        paragraph_agg_df['id'] = paragraph_agg_df.index
        paragraph_agg_df = paragraph_agg_df.reset_index(drop=True)
        paragraph_agg_df.drop(columns=["paragraph_word_count_count"], inplace=True)
        paragraph_agg_df = paragraph_agg_df.rename(columns={"paragraph_len_count": "paragraph_count"})
        return paragraph_agg_df

    def difficulty(self):
        df = copy.deepcopy(self.essays)
        df['token'] = [word_tokenize(p) for p in df["essay"]]
        df['token_len'] = df['token'].apply(lambda x: list(len(word) for word in x))
        df['verylong'] = df['token_len'].apply(lambda x: sum(c >= 9 for c in x))
        df['long'] = df['token_len'].apply(lambda x: sum(c == 7 or c == 8 for c in x))
        df['mid'] = df['token_len'].apply(lambda x: sum(c == 5 or c == 6 for c in x))
        df['difficulty'] = df['verylong'] * 5 + df['long'] * 3 + df['mid'] * 1
        df['long_words'] = df['verylong'] + df['long']
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'id'}, inplace=True)

        # sentence
        df_sentence = copy.deepcopy(self.essays)
        df_sentence['id'] = df_sentence.index
        # 避免将类似于i.e.切分成多个句子
        df_sentence['essay'] = df_sentence['essay'].apply(lambda x: re.sub(r'q\.q\.', 'qqq', x))
        df_sentence['sent'] = df_sentence['essay'].apply(lambda x: re.split('\\.|\\?|\\!', x))
        df_sentence = df_sentence.explode('sent')  # explode将列表里的元素展开
        df_sentence['sent'] = df_sentence['sent'].apply(lambda x: x.replace('\n', '').strip())  # strip会删除字符串两端的空格
        # Number of characters in sentences
        df_sentence['sent_len'] = df_sentence['sent'].apply(lambda x: len(x))
        df_sentence['sent_word_count'] = df_sentence['sent'].apply(lambda x: len(x.split(' ')))
        df_sentence = df_sentence[df_sentence['sent_len'] != 0]

        df_sentence['sentence_token'] = [word_tokenize(p) for p in df_sentence["sent"]]
        df_sentence['sentence_token_len'] = df_sentence['sentence_token'].apply(lambda x: list(len(word) for word in x))
        df_sentence['sentence_verylong'] = df_sentence['sentence_token_len'].apply(lambda x: sum(c >= 9 for c in x))
        df_sentence['sentence_long'] = df_sentence['sentence_token_len'].apply(
            lambda x: sum(c == 7 or c == 8 for c in x))
        df_sentence['sentence_mid'] = df_sentence['sentence_token_len'].apply(
            lambda x: sum(c == 5 or c == 6 for c in x))
        df_sentence['sentence_difficulty'] = df_sentence['sentence_verylong'] * 5 + df_sentence['sentence_long'] * 3 + \
                                             df_sentence['sentence_mid'] * 1
        df_sentence['sentence_long_words'] = df_sentence['sentence_verylong'] + df_sentence['sentence_long']
        a = df_sentence.groupby('id')[
            ['sentence_verylong', 'sentence_long', 'sentence_mid', 'sentence_difficulty', 'sentence_long_words']].agg(
            ['max', 'mean', 'sum'])
        a.columns = ['_'.join(x) for x in a.columns]

        return (df[['id', 'verylong', 'long', 'mid', 'difficulty', 'long_words']]).merge(a, on='id', how='left')

    def entropy(self):
        essay_df = copy.deepcopy(self.essays)
        essay_df['id'] = essay_df.index
        essay_df['essay'] = essay_df['essay'].apply(lambda x: re.sub(r'q\.q\.', 'qqq', x))
        essay_df['sent'] = essay_df['essay'].apply(lambda x: re.split('\\.|\\?|\\!', x))
        essay_df = essay_df.explode('sent')  # explode将列表里的元素展开
        essay_df['sent'] = essay_df['sent'].apply(lambda x: x.replace('\n', '').strip())  # strip会删除字符串两端的空格
        # Number of characters in sentences
        essay_df['sent_len'] = essay_df['sent'].apply(lambda x: len(x))
        essay_df['complexity'] = essay_df['sent'].apply(lambda x: calculate_entropy(x))
        a = essay_df.groupby('id').agg({'complexity': ['max', 'mean', 'std', 'sum', 'median']})
        a.columns = ['_'.join(i) for i in a.columns]
        return a.reset_index()

    def make_feats(self, df):
        feats = pd.DataFrame({'id': df['id'].unique().tolist()})

        print("Engineering time data")
        for gap in self.gaps:
            df[f'up_time_shift{gap}'] = df.groupby('id')['up_time'].shift(gap)
            df[f'action_time_gap{gap}'] = df['down_time'] - df[f'up_time_shift{gap}']

        df.drop(columns=[f'up_time_shift{gap}' for gap in self.gaps], inplace=True)

        print("Engineering cursor position data")
        for gap in self.gaps:
            df[f'cursor_position_shift{gap}'] = df.groupby('id')['cursor_position'].shift(gap)
            df[f'cursor_position_change{gap}'] = df['cursor_position'] - df[f'cursor_position_shift{gap}']
            df[f'cursor_position_abs_change{gap}'] = np.abs(df[f'cursor_position_change{gap}'])
        df.drop(columns=[f'cursor_position_shift{gap}' for gap in self.gaps], inplace=True)

        print("Engineering word count data")
        for gap in self.gaps:
            df[f'word_count_shift{gap}'] = df.groupby('id')['word_count'].shift(gap)
            df[f'word_count_change{gap}'] = df['word_count'] - df[f'word_count_shift{gap}']
            df[f'word_count_abs_change{gap}'] = np.abs(df[f'word_count_change{gap}'])
        df.drop(columns=[f'word_count_shift{gap}' for gap in self.gaps], inplace=True)

        print("Engineering statistical summaries for features")
        feats_stat = [
            ('event_id', ['max']),
            ('activity', ['nunique']),
            ('down_event', ['nunique']),
            ('up_event', ['nunique']),
            ('text_change', ['nunique']),
        ]
        for gap in self.gaps:
            feats_stat.extend([
                (f'action_time_gap{gap}',
                 ['max', 'min', 'mean', 'std', 'quantile', 'sem', 'sum', 'skew', pd.DataFrame.kurt]),
                (f'cursor_position_change{gap}',
                 ['max', 'mean', 'std', 'quantile', 'sem', 'sum', 'skew', pd.DataFrame.kurt]),
                (
                    f'word_count_change{gap}',
                    ['max', 'mean', 'std', 'quantile', 'sem', 'sum', 'skew', pd.DataFrame.kurt]),
            ])

        pbar = tqdm(feats_stat)
        for item in pbar:
            colname, methods = item[0], item[1]
            for method in methods:
                pbar.set_postfix()
                if isinstance(method, str):
                    method_name = method
                else:
                    method_name = method.__name__
                pbar.set_postfix(column=colname, method=method_name)
                tmp_df = df.groupby(['id']).agg({colname: method}).reset_index().rename(
                    columns={colname: f'{colname}_{method_name}'})
                feats = feats.merge(tmp_df, on='id', how='left')

        print("Engineering activity counts data")
        tmp_df = self.activity_counts(df)
        feats = pd.concat([feats, tmp_df], axis=1)

        print("Engineering event counts data")
        tmp_df = self.event_counts(df, 'down_event')
        feats = pd.concat([feats, tmp_df], axis=1)
        tmp_df = self.event_counts(df, 'up_event')
        feats = pd.concat([feats, tmp_df], axis=1)

        print("Engineering text change counts data")
        tmp_df = self.text_change_counts(df)
        feats = pd.concat([feats, tmp_df], axis=1)

        print("Engineering punctuation counts data")
        tmp_df = self.match_punctuations(df)
        feats = pd.concat([feats, tmp_df], axis=1)

        print("Engineering input words data")
        tmp_df = self.get_input_words(df)
        feats = pd.merge(feats, tmp_df, on='id', how='left')

        print("Calculating pause features")
        tmp_df = self.calculate_pauses(df)
        feats = pd.merge(feats, tmp_df, on='id', how='left')

        print('<merge brute force agg.>')
        tmp_df = self.brute_force_agg(df)
        feats = pd.merge(feats, tmp_df, on='id', how='left')

        print("Engineering ratios data")
        feats['word_time_ratio'] = feats['tmp_word_count_max'] / feats['tmp_down_time_max']
        feats['word_event_ratio'] = feats['tmp_word_count_max'] / feats['event_id_max']
        feats['event_time_ratio'] = feats['event_id_max'] / feats['tmp_down_time_max']
        feats['idle_time_ratio'] = feats['action_time_gap1_sum'] / feats['tmp_down_time_max']

        print('<merge duration_features.>')
        tmp_df = self.duration_features(df)
        feats = pd.merge(feats, tmp_df, on='id', how='left')
        if self.method == 'train':
            feats = feats.merge(self.train_scores, on='id', how='left')

        print('<merge countvectorizer_and_tfidf_features.>')
        if self.method == 'train':
            tmp_df, tokenizer, save_cols = self.essay_CountVectorizer_and_tfidf()
        else:
            tmp_df = self.essay_CountVectorizer_and_tfidf()
        feats = pd.merge(feats, tmp_df, on='id', how='left')

        print('<merge other features.>')
        if self.method == 'train':
            if os.path.exists('/kaggle/input/lgbm-and-nn-on-sentences'):
                tmp_df = pd.read_csv('/kaggle/input/lgbm-and-nn-on-sentences/train_agg_ratio.csv')
            else:
                tmp_df = self.other_features(df)
        else:
            tmp_df = self.other_features(df)
        feats = pd.merge(feats, tmp_df, on='id', how='left')

        print('<merge errors features.>')
        tmp_df = self.language_error(df)
        feats = feats.merge(tmp_df, on='id', how='left')
        tmp_df = self.sentence_error()
        feats = feats.merge(tmp_df, on='id', how='left')
        if self.method == 'train':
            tmp_df = self.language_error_letter()
        else:
            essays_upper = getEssays_with_upper(df)
            tmp_df = self.language_error_letter()
        feats = feats.merge(tmp_df, on='id', how='left')

        print('merge sentence and paragraph agg features')
        sent_df = self.split_essays_into_sentences()
        tmp_df = self.compute_sentence_aggregations(sent_df)
        feats = feats.merge(tmp_df, on='id', how='left')

        paragraph_df = self.split_essays_into_paragraphs()
        tmp_df = self.compute_paragraph_aggregations(paragraph_df)
        feats = feats.merge(tmp_df, on='id', how='left')

        print('merge R burst features')
        tmp_df = self.R_burst(df)
        feats = feats.merge(tmp_df, on='id', how='left')

        print('merge difficulty agg features')
        tmp_df = self.difficulty()
        feats = feats.merge(tmp_df, on='id', how='left')

        # print('merge sentence entropy features')
        # tmp_df = self.entropy()
        # feats = feats.merge(tmp_df, on='id', how='left')

        if self.method == 'train':
            return feats, tokenizer, save_cols
        else:
            return feats


def LGBM_train_and_test_v1(features, target_col, train_feats, test_feats, params):
    OOF_PREDS = np.zeros(len(train_feats))
    TEST_PREDS = np.zeros(len(test_feats))
    best_iters_dict = defaultdict(list)
    models_dict = {}
    scores = []
    test_predict_list = []
    best_params = params
    best_iterations = [340, 318, 325, 301, 361]
    for i in range(5):
        seeds = [3, 6, 38, 39, 43]
        seed = seeds[i]
        kf = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
        oof_valid_preds = np.zeros(train_feats.shape[0])
        X_test = test_feats
        for fold, (train_idx, valid_idx) in enumerate(kf.split(train_feats)):
            params = {
                "objective": "regression",
                "metric": "rmse",
                "random_state": 42,
                "n_estimators": best_iterations[i],
                "verbosity": -1,
                **best_params
            }

            X_train_pre, y_train_pre = train_feats.iloc[train_idx][features], train_feats.iloc[train_idx][target_col]
            X_valid_pre, y_valid_pre = train_feats.iloc[valid_idx][features], train_feats.iloc[valid_idx][target_col]
            pre_model = lgb.LGBMRegressor(**params)
            pre_model.fit(X_train_pre, y_train_pre, eval_set=[(X_valid_pre, y_valid_pre)], verbose=100)
            imp_df = pd.DataFrame()
            imp_df["feature"] = features
            imp_df["importance"] = pre_model.feature_importances_
            imp_df = imp_df.sort_values(by='importance', ascending=False)
            features_select = list(imp_df[imp_df['importance'] != 0]['feature'].values)
            print('-' * 50)

            X_train, y_train = train_feats.iloc[train_idx][features_select], train_feats.iloc[train_idx][target_col]
            X_valid, y_valid = train_feats.iloc[valid_idx][features_select], train_feats.iloc[valid_idx][target_col]
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=100)
            best_iters_dict[str(seed)].append(model.best_iteration_)
            valid_predict = model.predict(X_valid)
            oof_valid_preds[valid_idx] = valid_predict
            OOF_PREDS[valid_idx] += valid_predict / len(seeds)
            test_predict = model.predict(X_test[features_select])
            TEST_PREDS += test_predict / len(seeds) / 10
            test_predict_list.append(test_predict)
            score = metrics.mean_squared_error(y_valid, valid_predict, squared=False)
            models_dict[f'{fold}_{i}'] = model
        oof_score = metrics.mean_squared_error(train_feats[target_col], oof_valid_preds, squared=False)
        scores.append(oof_score)
    return OOF_PREDS, TEST_PREDS


def main():
    INPUT_DIR = "data"
    train_logs = pd.read_csv(f"{INPUT_DIR}/train_logs.csv")
    train_scores = pd.read_csv(f"{INPUT_DIR}/train_scores.csv")
    test_logs = pd.read_csv(f'{INPUT_DIR}/test_logs.csv')
    ss_df = pd.read_csv(f'{INPUT_DIR}/sample_submission.csv')
    train_essays = pd.read_csv(f"{INPUT_DIR}/data/train_essays_02.csv")
    train_essays.index = train_essays["Unnamed: 0"]
    train_essays.index.name = None
    train_essays.drop(columns=["Unnamed: 0"], inplace=True)
    train_essays_with_upper = pd.read_csv(f"{INPUT_DIR}/essays_with_upper.csv")
    train_essays_with_upper.index = train_essays_with_upper["Unnamed: 0"]
    train_essays_with_upper.index.name = None
    train_essays_with_upper.drop(columns=["Unnamed: 0"], inplace=True)

    train_logs = preprocessing(train_logs, dataset="train")
    test_logs = preprocessing(test_logs)

    preprocessor = Preprocessor_v1(42, train_essays, train_essays_with_upper, train_scores=train_scores)
    train_feats, tokenizer, save_cols = preprocessor.make_feats(train_logs)

    test_essays = EssayConstructor().getEssays(test_logs)
    test_essays.set_index('id', inplace=True)
    test_essays.index.name = None
    test_essays_with_upper = getEssays_with_upper(test_logs)
    preprocessor = Preprocessor_v1(42, test_essays, test_essays_with_upper, tokenizer=tokenizer, method='test',
                                   save_cols=save_cols)
    test_feats = preprocessor.make_feats(test_logs)

    target_col = ['score']
    f_read = open(f"{INPUT_DIR}/feats_dict.pkl", "rb")
    lgb_cols_v1 = pickle.load(f_read)
    f_read.close()
    print(f"lgb_cols_v1's length: {len(lgb_cols_v1)}")

    params1 = {'boosting_type': 'gbdt',
               'metric': 'rmse',
               'reg_alpha': 0.003188447814669599,
               'reg_lambda': 0.0010228604507564066,
               'colsample_bytree': 0.5420247656839267,
               'subsample': 0.9778252382803456,
               'feature_fraction': 0.8,
               'bagging_freq': 1,
               'bagging_fraction': 0.75,
               'num_leaves': 19,
               'learning_rate': 0.01716485155812008,
               'min_child_samples': 46}

    OOF_PREDS_v1, TEST_PREDS_v1 = LGBM_train_and_test_v1(lgb_cols_v1, target_col, train_feats, test_feats, params1)

    print('OOF metric LGBM v1 = {:.5f}'.format(
        metrics.mean_squared_error(train_feats[target_col], OOF_PREDS_v1, squared=False)))


if __name__ == "__main__":
    main()
