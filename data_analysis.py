from pyspark import SparkContext,SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import Row
import pyspark
import time
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, LongType, DoubleType, IntegerType
from pyspark.sql.types import FloatType
from pyspark.ml import classification
from pyspark.ml.feature import VectorAssembler
from datetime import datetime
from pyspark.sql import functions as f
import pandas as pd
import numpy as np

def get_data(sc, path, cov_event, **kwargs):
    try:
        raw_data = sc.textFile(input_file)
    except:
        print ('load data failed, try again')
        return (-1)
    data_rdd = raw_data.map(lambda x:x.split(',')[1].split(':'))
    #if not data_rdd.filter(lambda x:cov_event in x).count():
    #    print ('No cov events, check again')
    #    return (-1)
    return (data_rdd)

def data_description(data_rdd, cov_event):
    print ('Total records:', data_rdd.count())
    print ('Total conversions(event {}): {}'.format(cov_event, data_rdd.filter(lambda x:cov_event in x).count()))
    print ('Avg event length: {0}/Std event length: {1}'.format(data_rdd.map(lambda x:len(x)).mean(), 
                                                                data_rdd.map(lambda x:len(x)).sampleStdev()))
    print ('Avg conv pos: {0}/Std conv pos: {1}'.format(data_rdd.filter(lambda x:cov_event in x).map(lambda x:x.index(cov_event)).mean(),
                                                        data_rdd.filter(lambda x:cov_event in x).map(lambda x:x.index(cov_event)).sampleStdev()))
    return()

def get_events_from_data(data_rdd, cov_event, keep = 'last'):
    all_events = data_rdd.map(lambda x:cut_tail(x, cov_event)).reduce(lambda x, y:list(set(x + y)))
    all_events = list(set(all_events).difference(set([cov_event])))
    #return (all_events)
    if keep == 'last':
        return (all_events, all_events)
    elif isinstance(keep, int):
        first_events = data_rdd.map(lambda x:x[:keep]).reduce(lambda x, y: list(set(x + y)))
        first_events = list(set(first_events).difference(set([cov_event])))
        return (all_events, first_events)

def event_map(seq, cut_position, all_events, label):
    seq_map = {}
    if cut_position:
        for event in all_events:
            seq_map[event] = sum([1 if x == event else 0 for x in seq[:cut_position]])
    else:
        for event in all_events:
            seq_map[event] = sum([1 if x == event else 0 for x in seq])
    if label:
        if isinstance(label, str):
            seq_map['label'] = 1 if sum([1 if x == label else 0 for x in seq])>0 else 0
        elif isinstance(label, list):
            seq_map['label'] = 1 if sum([1 if x in label else 0 for x in seq])>0 else 0
    return(seq_map)

def cut_tail(lst, cut_event): ## Not tested
    new_lst = []
    if cut_event not in lst:
        return (lst)
    else:
        for event in lst:
            if event == cut_event:
                new_lst.append(event)
                break
            else:
                new_lst.append(event)
        #new_lst.append([1053]) ## to avoid void return list
        return (new_lst)

def event_seq_map(seq, cut_position, all_event_seqs, label):
    seq_map = {}
    if cut_position:
        for event_seq in all_event_seqs:
            prev_event_idx = 0
            seq_map[str(event_seq)] = 1
            for event in event_seq:
                if event not in seq[:cut_position]:
                    seq_map[str(event_seq)] = 0
                    break
                else:
                    event_idx = seq[:cut_position].index(event)
                    if event_idx < prev_event_idx:
                        seq_map[str(event_seq)] = 0
                        break
                    else:
                        prev_event_idx = event_idx
    else:
        for event_seq in all_event_seqs:
            prev_event_idx = 0
            seq_map[str(event_seq)] = 1
            for event in event_seq:
                if event not in seq:
                    seq_map[str(event_seq)] = 0
                    break
                else:
                    event_idx = seq.index(event)
                    if event_idx < prev_event_idx:
                        seq_map[str(event_seq)] = 0
                        break
                    else:
                        prev_event_idx = event_idx
    if label:
        if isinstance(label, str):
            seq_map['label'] = 1 if sum([1 if x == label else 0 for x in seq])>0 else 0
        elif isinstance(label, list):
            seq_map['label'] = 1 if sum([1 if x in label else 0 for x in seq])>0 else 0
    return(seq_map)

def rdd2df(data_rdd, feature_events, cov_event):
    data_df = data_rdd.filter(lambda x:x[0]!= cov_event)
    data_df = data_df.map(lambda x:cut_tail(x, cov_event))
    data_df = data_df.map(lambda x:Row(**event_map(x, None, feature_events, cov_event))).toDF()
    return data_df

def feature_event_filter(df, feature_events, cov_event):
    all_corr = []
    print ('All_events:', feature_events)
    df.cache()
    for event in feature_events:
        all_corr.append([event, df.stat.corr(str(event), 'label')])
    return all_corr

def model_build(df, feature_events, cov_event, model_name = 'logistic'):
    df_train, df_test = df.randomSplit([0.7, 0.3], 23333)
    va = VectorAssembler(inputCols= feature_events, outputCol='features')
    df_train_converted = va.transform(df_train)
    df_test_converted = va.transform(df_test)
    if model_name == 'logistic':
        clf = classification.LogisticRegression(regParam = 0.01, labelCol = 'label')
    model = clf.fit(df_train_converted)
    train_result = model.evaluate(df_train_converted)
    test_result = model.evaluate(df_test_converted)
    print (train_result.areaUnderROC)
    print (test_result.areaUnderROC)
    fi = pd.DataFrame(zip(feature_events, model.coefficients.toArray()),columns = ['feature', 'importance'])
    print (fi.sort_values(by = 'importance', ascending = False))
    return (model, va, df_train, df_test)

def model_inference(df, model, va, feature_events, cov_event, ref_event = None, ref_target = 0):
    if ref_event:
        df_inference = df.withColumn(ref_event, f.lit(ref_target))
    else:
        df_inference = df
    df_inference = va.transform(df_inference)
    pred = model.transform(df_inference)
    return (pred.groupby().agg(f.sum(f.udf(lambda x:np.float(x[1]), FloatType())('probability'))).toPandas())

if __name__=='__main__':
    input_file = 'hdfs://wdc04np-datascience-01.adm01.com:7120//user/zhengzhe1/tl_ea/amex_events.txt'
    cov_event = '1060'
    sc = pyspark.SparkContext()
    spark = SparkSession(sc)
    data_rdd = get_data(sc, input_file, cov_event)
    data_description(data_rdd, cov_event)
    all_events, events = get_events_from_data(data_rdd, cov_event, keep = 'last')
    df = rdd2df(data_rdd, events, cov_event)
    all_corr = feature_event_filter(df, events, cov_event)
    to_drop_events = [event_meta[0] for event_meta in all_corr if abs(event_meta[1])>0.9]
    filtered_feature_events = list(set(events).difference(to_drop_events))
    model, va, df_train, df_test = model_build(df, filtered_feature_events, cov_event)
    base_conv = df_test.groupby().agg(f.sum('label')).toPandas().values.ravel()[0]
    pred_conv = model_inference(df_test, model, va, filtered_feature_events, 
                                cov_event, ref_event=None, ref_target = 0)
    hypo_conv = model_inference(df_test, model, va, filtered_feature_events, 
                                cov_event, ref_event='1178', ref_target = 0)
    print ('base_conv:', base_conv)
    print ('pred_conv:', pred_conv)
    print ('hypo_conv:', hypo_conv)

