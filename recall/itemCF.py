#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os, sys
from pyspark.sql import SparkSession, Row
from pyspark.sql.types import *
from subprocess import call
from optparse import OptionParser
import datetime as dt
import time
import json
import math
import random
from heapq import merge
from itertools import islice

is_debug = True
poster_limit = 5
max_user_items = 100000
# minimum number of items clicked/operated by user, user who have less behavior will be filtered
# min_user_items = 15
# min_user_items = 6
min_user_items = 2
# window size
# window_size = 100
window_size = 10000
# after merge, if the weight of concurrency is smaller than this threshold
# min_weight_tld = 2.25
min_weight_tld = 3
# min_weight_tld = 5.25
# maximum number of similar items;
max_item_num_s1 = 1000
# maximum number of similar items:
max_item_num_s2 = 500
# maximum number of similar items;
max_item_num_s3 = 200
# concurrency weight threshold
min_score_tld = 0.001
# 每个item最少保留的相似视频数，如果小于这值，就不看score阈值
# the minimum number of similar items, if the number of similar items is less than this value, score will not be considered
min_item_num = 50
# average positive behavior rate
avg_pos_rate = 0.035
# smooth play cnt
play_cnt_smooth = 100
min_pos_rate_ratio = 0
# the weight of like counter
like_count_weight = 0.45
# the weight of share behavior
share_count_weight = 0.55


def usage():
    parser = OptionParser()
    parser.add_option('--day', dest="day", default="")
    parser.add_option('--hour', dest="hour", default="")
    parser.add_option('--region', dest="region", default="ID")
    parser.add_option('--post_days', dest="post_days", default="30")
    parser.add_option('--merge_days', dest="merge_days", default="30")
    parser.add_option('--action', dest="action", default="gen_data")
    parser.add_option('--partition', dest="partition", default="5000")
    parser.add_option('--executors', dest="executors", default="500")
    parser.add_option('--input', dest='input', default='')
    parser.add_option('--output', dest='output', default='')
    parser.add_option('--pos_count_tld', dest='pos_count_tld', default='80')
    parser.add_option('--final_score_tld', dest='final_score_tld', default='0.01')
    parser.add_option('--final_item_tld', dest='final_item_tld', default='200')
    parser.add_option('--final_min_tld', dest='final_min_tld', default='50')
    parser.add_option('--base_path', dest='base_path',default='your hdfs path')
    parser.add_option('--rcmdble_table_path', dest='rcmdble_table_path',default='your recommendable table path')
    (options, _) = parser.parse_args()
    return options


def get_recommendable_items(spark, options):
    region = options.region
    date = options.date

    sql_cmd = '''
    select itemid from rcmd_feature.dd_recommendable_items where region = '{}' and date= DATE'{}'
    '''.format(region, date)

    if is_debug:
        print("[DEBUG] get_recommendable_items sql is: %s" % sql_cmd)
    df = spark.sql(sql_cmd)
    rows = df.collect()
    recommendable_item_set = set()
    for row in rows:
        recommendable_item_set.add(row.itemid)
    if is_debug:
        print("[DEBUG] len of recommendable items set is %d" % len(recommendable_item_set))
    b_item_set = spark.sparkContext.broadcast(recommendable_item_set)
    return b_item_set

# 解析每行数据
def extractClickItemData(row):
    detail_obj = json.loads(row.data)
    str_itemid = str(detail_obj.get("itemid", -1))
    itemid = 0
    if str_itemid.isdigit():
        itemid = int(str_itemid)
    #   shopid = int(detail_obj.get("shopid", -1))
    action_weight = 1.0
    return row.userid, [(-action_weight, itemid, row.event_timestamp)]

# 统计每天数据中(vi,vj) pair对的共现次数
def concurrence_stat_day(spark, options):
    def stat_map(line):
        (_, values) = line
        values.sort(key=lambda i: i[2])  # 根据时间排序
        for i in range(0, len(values)):
            vi = values[i][1]
            scorei = -values[i][0]
            yield ((vi, 0), scorei * scorei)  # 为了统计每个item的出现次数
            for j in range(i + 1, min(i + 1 + window_size, len(values))):
                vj = values[j][1]
                scorej = -values[j][0]
                if vi == vj:
                    continue
                if vi < vj:
                    yield ((vi, vj), scorei * scorej)
                else:
                    yield ((vj, vi), scorei * scorej)

    day = options.day
    output_path = "%s/%s/day_data/%s" % (options.base_path, options.region, day)

    sql_cmd = '''
        SELECT userid, event_timestamp, operation, data 
        FROM traffic.shopee_traffic_dwd_click_hi__reg_s1_live
        WHERE page_type='home' and page_section[0] = 'daily_discover' 
        AND data is not NULL and length(data) > 0
        AND target_type='item' and operation='click' 
        AND date='{}' and userid > 0 and region='{}'
    '''.format(options.day, options.region)

    df = spark.sql(sql_cmd)
    partition = int(options.partition)
    # extractClickItemData => row.userid, [(-action_weight, itemid, row.event_timestamp)]
    df = df.rdd.repartition(partition).map(extractClickItemData) \
        .filter(lambda t: t[1][0][1] != 0) \
        .reduceByKey(lambda x, y: list(islice(merge(x, y), max_user_items)), partition) \
        .filter(lambda t: len(t[1]) >= min_user_items) \
        .flatMap(stat_map)   \
        .reduceByKey(lambda x, y: x + y, partition) \
        .map(lambda t: Row(vi=t[0][0], vj=t[0][1], weight=t[1])) \
        .toDF()

    call("hadoop fs -rmr %s" % output_path, shell=True)
    df.repartition(100).write.format("parquet").mode("overwrite").save(output_path)
    return output_path


def remove_day_data(spark, options):
    keep_num = 62
    hour_path = "%s/%s/day_data" % (options.base_path, options.region)
    cmd_str = "hadoop fs -ls %s/*/_SUCCESS | awk '{{if(NF==8)print $NF}}' | sort -r" % hour_path
    cmd_res = os.popen(cmd_str).readlines()
    remove_data = cmd_res[keep_num:]
    for filename in remove_data:
        filename = filename.strip('\r\n')
        if filename.endswith("_SUCCESS"):
            filename = filename[:-len("_SUCCESS")]
        call("hadoop fs -rmr %s" % filename, shell=True)


def remove_merge_data(spark, options, is_group=False):
    keep_num = 10
    if is_group:
        hour_path = "%s/%s/group_merge_data" % (options.base_path, options.region)
    else:
        if int(options.merge_days) != 30:
            hour_path = "%s/%s/merge_data_%s" % (options.base_path, options.region, options.merge_days)
        else:
            hour_path = "%s/%s/merge_data" % (options.base_path, options.region)

    cmd_str = "hadoop fs -ls %s/*/_SUCCESS | awk '{{if(NF==8)print $NF}}' | sort -r" % hour_path
    cmd_res = os.popen(cmd_str).readlines()
    remove_data = cmd_res[keep_num:]
    for filename in remove_data:
        filename = filename.strip('\r\n')
        if filename.endswith("_SUCCESS"):
            filename = filename[:-len("_SUCCESS")]
        call("hadoop fs -rmr %s" % filename, shell=True)


def get_last_merge_data(merge_base):
    cmd_str = "hadoop fs -ls %s/*/_SUCCESS | awk '{{if(NF==8)print $NF}}' | sort -r" % merge_base
    cmd_res = os.popen(cmd_str).readlines()
    if len(cmd_res) > 0:
        last_file = cmd_res[0].strip('\r\n')
        if last_file.endswith("_SUCCESS"):
            return last_file[:-len("_SUCCESS")]


def get_concurrence_table(spark, df):
    concurrent_df = df.filter(df.vb != 0)
    concurrent_df.createOrReplaceTempView("concurrence_table")

    # 统计vi出现次数
    item_df = df.filter(df.vb == 0).select(df.va.alias("vid"), df.weight.alias("weight"))
    item_df.createOrReplaceTempView("vid_table_tmp")

    sql_cmd = '''
    select t1.vid as vid,
        t1.weight as weight 
    from vid_table_tmp t1
    '''
    df = spark.sql(sql_cmd)
    df.createOrReplaceTempView("vid_table")


def join_cnt(spark, options, is_group=False):
    if is_group:
        merge_base = "%s/%s/group_merge_data" % (options.base_path, options.region)
    else:
        if int(options.merge_days) != 30:
            merge_base = "%s/%s/merge_data_%s" % (options.base_path, options.region, options.merge_days)
        else:
            merge_base = "%s/%s/merge_data" % (options.base_path, options.region)

    merge_path = get_last_merge_data(merge_base)
    if merge_path == None:
        print("merge data is not ready, path: %s" % merge_path)
        sys.exit(-1)
    if is_debug:
        print("[DEBUG] merge_path: %s" % merge_path)
    df = spark.read.parquet(merge_path)
    get_concurrence_table(spark, df)
    sql_cmd = '''
    select t1.va as va,
        t1.vb as vb,
        t1.weight as weight,
        t2.weight as weight_a,
        t3.weight as weight_b
    from concurrence_table t1
    join vid_table t2 on t1.va = t2.vid
    join vid_table t3 on t1.vb = t3.vid
    '''
    df = spark.sql(sql_cmd)
    if is_group:
        output_path = "%s/%s/group_join_cnt" % (options.base_path, options.region)
    else:
        if int(options.merge_days) != 30:
            output_path = "%s/%s/join_cnt_%s" % (options.base_path, options.region, options.merge_days)
        else:
            output_path = "%s/%s/join_cnt" % (options.base_path, options.region)

    call("hadoop fs -rmr %s" % output_path, shell=True)
    df.repartition(1000).write.format("parquet").mode("overwrite").save(output_path)


def group_data(spark, options, is_group=False):
    def group_map(row):
        score = -float(row.weight) / math.sqrt(row.weight_a) / math.sqrt(row.weight_b)
        yield (row.va, [(score, row.vb, (row.weight, row.weight_a, row.weight_b))])
        yield (row.vb, [(score, row.va, (row.weight, row.weight_b, row.weight_a))])

    def trunc_map(values):
        poster_dedup_dict = {}
        res = []
        for value in values:
            vid = value[1]
            poster_uid = vid & 4294967295
            poster_cnt = poster_dedup_dict.get(poster_uid, 0)
            if poster_cnt >= poster_limit:
                continue
            poster_dedup_dict[poster_uid] = poster_cnt + 1
            res.append(value)
        if len(res) > max_item_num_s2:
            res = res[:max_item_num_s2]
        if len(res) == 0:
            return None
        return res

    def trunc_map_v2(row):
        res = []
        for i in range(0, min(len(row.value), max_item_num_s3)):
            value = [row.value[i][0], row.value[i][1]]
            value[0] = -value[0]
            if value[0] > 1.0:
                value[0] = 1.0
            if len(res) < min_item_num or value[0] > min_score_tld:
                res.append(value)
            else:
                break
        return (row.key, res)

    if is_group:
        input_path = "%s/%s/group_join_cnt" % (options.base_path, options.region)
    else:
        if int(options.merge_days) != 30:
            input_path = "%s/%s/join_cnt_%s" % (options.base_path, options.region, options.merge_days)
        else:
            input_path = "%s/%s/join_cnt" % (options.base_path, options.region)

    df = spark.read.parquet(input_path).rdd \
        .flatMap(group_map) \
        .repartition(int(options.partition)) \
        .reduceByKey(lambda x, y: list(islice(merge(x, y), max_item_num_s1)), int(options.partition)) \
        .filter(lambda t: t[1] != None) \
        .map(lambda t: Row(key=t[0], value=t[1])) \
        .toDF()

    if is_group:
        output_path = "%s/%s/group_group_data" % (options.base_path, options.region)
    else:
        if int(options.merge_days) != 30:
            output_path = "%s/%s/group_data_%s" % (options.base_path, options.region, options.merge_days)
        else:
            output_path = "%s/%s/group_data" % (options.base_path, options.region)

    call("hadoop fs -rmr %s" % output_path, shell=True)
    df.repartition(1000).write.format("parquet").mode("overwrite").save(output_path)

    rdd = spark.read.parquet(output_path).rdd \
        .map(trunc_map_v2) \
        .filter(lambda t: len(t[1]) > 0) \
        .map(lambda t: "%s\t%s" % (t[0], ";".join(["%s:%f" % (i[1], i[0]) for i in t[1]])))
    if is_group:
        output_path = "%s/%s/group_final_data" % (options.base_path, options.region)
    else:
        if int(options.merge_days) != 30:
            output_path = "%s/%s/final_data_%s" % (options.base_path, options.region, options.merge_days)
        else:
            output_path = "%s/%s/final_data" % (options.base_path, options.region)

    call("hadoop fs -rmr %s" % output_path, shell=True)
    rdd.repartition(100).saveAsTextFile(output_path, "org.apache.hadoop.io.compress.GzipCodec")


def merge_result(spark, options):
    merge_days = int(options.merge_days)
    partition = int(options.partition)
    if merge_days != 30:
        merge_base = "%s/%s/merge_data_%s" % (options.base_path, options.region, options.merge_days)
    else:
        merge_base = "%s/%s/merge_data" % (options.base_path, options.region)
    files = []

    for i in range(0, merge_days + 1):
        day = (dt.datetime.strptime(options.day, '%Y-%m-%d') - dt.timedelta(days=i)).strftime("%Y-%m-%d")
        filename = "%s/%s/day_data/%s" % (options.base_path, options.region, day)
        files.append(filename)

    if is_debug:
        print("[debug] files: %s" % len(files))

    b_recommendable_items = get_recommendable_items(spark, options)
    rdds = []
    for filename in files:
        rdd = spark.read.parquet(filename).rdd \
            .filter(lambda row: (row.vi in b_recommendable_items.value) and (
                    row.vj == 0 or (row.vj in b_recommendable_items.value))) \
            .map(lambda row: ((row.vi, row.vj), row.weight))
        rdds.append(rdd)
    df = spark.sparkContext.union(rdds) \
        .reduceByKey(lambda x, y: x + y, partition) \
        .filter(lambda t: t[1] >= min_weight_tld) \
        .map(lambda t: Row(va=t[0][0], vb=t[0][1], weight=t[1])) \
        .toDF()

    output_path = "%s/%s" % (merge_base, int(time.time()))
    df.repartition(1000).write.format("parquet").mode("overwrite").save(output_path)


def main():
    options = usage()

    # start of main program
    spark = SparkSession. \
        builder \
        .appName("icf_model") \
        .config("spark.sql.maxMetadataStringLength", 100000) \
        .config("spark.sql.maxToStringFields", 100000) \
        .config("spark.sql.debug.maxToStringFields", 100000) \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("hive.exec.dynamic.partition", "true") \
        .config("hive.exec.dynamic.partition.mode", "nonstrict") \
        .config("spark.sql.sources.partitionOverwriteMode", "DYNAMIC") \
        .config("mapreduce.fileoutputcommitter.marksuccessfuljobs", "true") \
        .config("spark.eventLog.enabled", "true") \
        .enableHiveSupport() \
        .getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')

    if options.action == "gen_data":
        concurrence_stat_day(spark, options)
        remove_day_data(spark, options)
    elif options.action == "merge_data":
        merge_result(spark, options)
        remove_merge_data(spark, options)
    elif options.action == "join":
        join_cnt(spark, options)
        group_data(spark, options)


if __name__ == '__main__':
    main()