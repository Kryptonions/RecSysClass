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
from pyspark.sql import functions as F
import subprocess


def usage():
    parser = OptionParser()
    parser.add_option('--day', dest="day", default="")
    # parser.add_option('--hour', dest = "hour", default = "")
    parser.add_option('--region', dest="region", default="ID")
    parser.add_option('--min_weight_tld', dest="min_weight_tld", default=0)
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
    parser.add_option('--edge_thre', dest='edge_thre', default='3')
    parser.add_option('--hip1_num', dest='hip1_num', default='10')
    parser.add_option('--hip2_num', dest='hip2_num', default='10')
    parser.add_option('--base_path', dest='base_path',
                      default='hdfs://R2/projects/rcmd_feature/hdfs/prod/alg/dd/model/graphsage_v0')
    parser.add_option('--csv_path', dest='csv_path',
                      default='hdfs://R2/projects/rcmd_feature/hdfs/prod/alg/dd/train_data/graphsage_v0')
    parser.add_option('--rcmdble_table_path', dest='rcmdble_table_path',
                      default='hdfs://R2/projects/rcmd_feature/hive/rcmd_feature/dd_recommendable_items')
    (options, _) = parser.parse_args()
    return options


def day_data(spark):
    output_path = f"{base_path}/{region}/day_data/{day}"
    dep_start_date = dt.datetime.strptime(day, '%Y-%m-%d') - dt.timedelta(days=1)
    dep_end_date = dt.datetime.strptime(day, '%Y-%m-%d') - dt.timedelta(days=0)
    dep_start_date_str = dep_start_date.strftime('%Y-%m-%d')
    dep_end_date_str = dep_end_date.strftime('%Y-%m-%d')
    if REGIONS_TIMEZONE[region] < 0:
        end_hour = 15
    else:
        end_hour = 23 - REGIONS_TIMEZONE[region]

    start_hour = end_hour + 1

    sql_cmd = f"""
    select
        session_id,
        CAST(get_json_object(data, "$.itemid") as BIGINT) as itemid,
        event_timestamp
    from
        traffic.shopee_traffic_dwd_click_hi__reg_s1_live
    where
        grass_region="{region}"
        AND ((utc_date=date'{dep_end_date_str}' and hour <= {end_hour}) or (utc_date = DATE'{dep_start_date_str}' and hour >= {start_hour}))
        and operation = "click" and target_type="item"
        AND data is not NULL and length(data) > 0
        AND userid > 0
    """
    df = spark.sql(sql_cmd)

    def gen_pair(row):
        itemids = row.itemids
        ts = row.ts
        itemids = [x[0] for x in sorted(zip(itemids, ts), key=lambda x: x[1])]
        for va, vb in zip(itemids[:-1], itemids[1:]):
            if va != vb:
                yield (va, vb, 1.0)
                yield (vb, va, 1.0)

    item_pair = df.na.drop().groupby("session_id").agg(F.collect_list("itemid").alias('itemids'),
                                                       F.collect_list("event_timestamp").alias('ts')) \
        .rdd.flatMap(gen_pair).toDF(["va", "vb", "w"]) \
        .groupBy("va", "vb").agg(F.sum("w").alias('w'))
    call("/usr/share/hadoop-client/bin/hadoop fs -rmr %s" % output_path, shell=True)
    item_pair.repartition(100).write.format("parquet").mode("overwrite").save(output_path)


def remove_day_data(spark):
    keep_num = 1000
    hour_path = f"{base_path}/{region}/day_data"
    cmd_str = "/usr/share/hadoop-client/bin/hadoop fs -ls %s/*/_SUCCESS | awk '{{if(NF==8)print $NF}}' | sort -r" % hour_path
    cmd_res = os.popen(cmd_str).readlines()
    remove_data = cmd_res[keep_num:]
    for filename in remove_data:
        filename = filename.strip('\r\n')
        if filename.endswith("_SUCCESS"):
            filename = filename[:-len("_SUCCESS")]
        call("/usr/share/hadoop-client/bin/hadoop fs -rmr %s" % filename, shell=True)


def get_recommendable_items(spark):
    cmd_str = "/usr/share/hadoop-client/bin/hadoop fs  -ls %s/grass_region=%s/*/part-00001* | awk '{if(NF==8) print $NF}' | sort -r | awk -F '=' '{print $3}' | awk -F '/' '{print $1}'" % (
    rcmdble_table_path, region)
    db_dates = os.popen(cmd_str).readlines()
    db_date_str = db_dates[0]
    db_date_str = db_date_str.strip('\r\n')
    sql_cmd = '''
    select itemid from rcmd_feature.dd_recommendable_items where grass_region = '{}' and grass_date= DATE'{}'
    '''.format(region, db_date_str)

    df = spark.sql(sql_cmd)
    rows = df.collect()
    recommendable_item_set = set()
    for row in rows:
        recommendable_item_set.add(row.itemid)
    b_item_set = spark.sparkContext.broadcast(recommendable_item_set)
    return b_item_set


def merge_result(spark):
    if merge_days != 30:
        merge_base = "%s/%s/merge_data_%s" % (base_path, region, merge_days)
    else:
        merge_base = "%s/%s/merge_data" % (base_path, region)
    files = []
    for i in range(0, merge_days + 1):
        merge_day = (dt.datetime.strptime(day, '%Y-%m-%d') - dt.timedelta(days=i)).strftime("%Y-%m-%d")
        filename = "%s/%s/day_data/%s" % (base_path, region, merge_day)
        files.append(filename)

    b_recommendable_items = get_recommendable_items(spark)
    rdds = []
    for filename in files:
        rdd = spark.read.parquet(filename).rdd \
            .filter(lambda row: (row.va in b_recommendable_items.value) and (row.vb in b_recommendable_items.value)) \
            .map(lambda row: ((row.va, row.vb), row.w))
        rdds.append(rdd)
    df = spark.sparkContext.union(rdds) \
        .reduceByKey(lambda x, y: x + y) \
        .filter(lambda t: t[1] >= min_weight_tld) \
        .map(lambda t: Row(va=t[0][0], vb=t[0][1], w=t[1])) \
        .toDF()

    output_path = "%s/%s" % (merge_base, int(time.time()))
    df.count()
    df.repartition(1000).write.format("parquet").mode("overwrite").save(output_path)


def remove_merge_data(spark):
    keep_num = 10

    if int(merge_days) != 30:
        hour_path = "%s/%s/merge_data_%s" % (base_path, region, merge_days)
    else:
        hour_path = "%s/%s/merge_data" % (base_path, region)

    cmd_str = "/usr/share/hadoop-client/bin/hadoop fs -ls %s/*/_SUCCESS | awk '{{if(NF==8)print $NF}}' | sort -r" % hour_path
    cmd_res = os.popen(cmd_str).readlines()
    remove_data = cmd_res[keep_num:]
    for filename in remove_data:
        filename = filename.strip('\r\n')
        if filename.endswith("_SUCCESS"):
            filename = filename[:-len("_SUCCESS")]
        call("/usr/share/hadoop-client/bin/hadoop fs -rmr %s" % filename, shell=True)


def get_last_merge_data(merge_base):
    cmd_str = "/usr/share/hadoop-client/bin/hadoop fs -ls %s/*/_SUCCESS | awk '{{if(NF==8)print $NF}}' | sort -r" % merge_base
    cmd_res = os.popen(cmd_str).readlines()
    if len(cmd_res) > 0:
        last_file = cmd_res[0].strip('\r\n')
        if last_file.endswith("_SUCCESS"):
            return last_file[:-len("_SUCCESS")]


def gen_2hop(label_df, df, spark):
    def sample_item(row, k=hip1_num):
        #     for p, b in enumerate(row.vbs):
        for t in range(int(row.w_day)):
            # tag = str(row.va) + '_' + str(row.vb_day) + '_' +str(t)
            tag = t
            for i in range(k // len(row.ws)):
                for vb in row.vbs:
                    yield (tag, row.va, row.vb_day, vb)

            sub_sampled_vbs = random.sample(row.vbs, k=k % len(row.ws))
            for vb in sub_sampled_vbs:
                yield (tag, row.va, row.vb_day, vb)

    df = df.groupBy('va').agg(F.collect_list("vb").alias('vbs'), F.collect_list("w").alias('ws'))

    day_sample = label_df.join(df, df.va == label_df.va_day)
    hop1_sample1 = day_sample \
        .rdd.flatMap(sample_item).toDF(['tag', 'va', 'vb', 'hop1'])

    def sample_list_rm_va(row, choices_method='u', k=hip2_num):
        center = row.va
        p = row.vbs.index(center)
        row.vbs.pop(p)
        row.ws.pop(p)
        if len(row.vbs) > 0:
            if choices_method == 'u':
                sampled_vbs = []
                for i in range(k // len(row.ws)):
                    sampled_vbs += row.vbs

                sub_sampled_vbs = random.sample(row.vbs, k=k % len(row.ws))
                sampled_vbs += sub_sampled_vbs

            return (row.tag, row.va, row.vb, row.hop1, sampled_vbs)
        else:
            return (row.tag, row.va, row.vb, row.hop1, [])

    hop2 = hop1_sample1.join(df, hop1_sample1.hop1 == df.va) \
        .select(hop1_sample1.tag, hop1_sample1.va, hop1_sample1.vb, hop1_sample1.hop1, df.vbs, df.ws) \
        .toDF(*['tag', 'va', 'vb', 'hop1', 'vbs', 'ws']) \
        .rdd.map(sample_list_rm_va).toDF(['tag', 'va', 'vb', 'hop1', 'hop2'])
    # hop2 = hop2.toDF(*['tag','va','vb','hop1','hop2'])

    res = hop2.groupBy('tag', 'va', 'vb').agg(F.collect_list("hop1").alias('hop1s'),
                                              F.collect_list("hop2").alias('hop2s'))
    # res.show(truncate = False)
    return res


def gen_va_hop(spark):
    day_path = f"{base_path}/{region}/day_data/{day}"
    merge_base = "%s/%s/merge_data" % (base_path, region)
    # output_path = f"{base_path}/{region}/train_data/{day}"
    print(merge_base)
    merge_path = get_last_merge_data(merge_base)
    df = spark.read.parquet(merge_path)
    df = df.filter(df.w > edge_thre)

    va_hop_path = f"{base_path}/{region}/va_hop/{day}"
    label_df = spark.read.parquet(day_path).toDF(*['va_day', 'vb_day', 'w_day'])
    va_hop = gen_2hop(label_df, df, spark).toDF(*['tag', 'va', 'vb', 'va_hop1', 'va_hop2'])
    va_hop.repartition(100).write.format("parquet").mode("overwrite").save(va_hop_path)


def gen_vb_hop(spark):
    day_path = f"{base_path}/{region}/day_data/{day}"
    merge_base = "%s/%s/merge_data" % (base_path, region)
    # output_path = f"{base_path}/{region}/train_data/{day}"
    merge_path = get_last_merge_data(merge_base)
    df = spark.read.parquet(merge_path)
    df = df.filter(df.w > edge_thre)

    vb_hop_path = f"{base_path}/{region}/vb_hop/{day}"
    label_df = spark.read.parquet(day_path).toDF(*['vb_day', 'va_day', 'w_day'])
    vb_hop = gen_2hop(label_df, df, spark).toDF(*['tag', 'vb', 'va', 'vb_hop1', 'vb_hop2'])
    vb_hop.repartition(100).write.format("parquet").mode("overwrite").save(vb_hop_path)


def convert_arr_hops(spark, train_data):
    def array_to_string(my_list):
        return '[' + ','.join([str(elem) for elem in my_list]) + ']'

    def arr_of_arr_to_string(my_list):
        res = "["
        for elem_arr in my_list:
            res += '[' + ','.join([str(elem) for elem in elem_arr]) + '],'
        res = res[:-1]
        res += "]"
        return res

    array_to_string_udf = F.udf(array_to_string, StringType())
    arr_of_arr_to_string_udf = F.udf(arr_of_arr_to_string, StringType())

    res_df = train_data.withColumn('va_hop1', array_to_string_udf(train_data["va_hop1"])) \
        .withColumn('vb_hop1', array_to_string_udf(train_data["vb_hop1"])) \
        .withColumn('va_hop2', arr_of_arr_to_string_udf(train_data["va_hop2"])) \
        .withColumn('vb_hop2', arr_of_arr_to_string_udf(train_data["vb_hop2"]))
    return res_df


def gen_train_sample(spark):
    # day_path = f"{base_path}/{region}/day_data/{day}"
    # merge_base = "%s/%s/merge_data" % (base_path, region)
    output_path = f"{base_path}/{region}/train_data/{day}"
    # hdfs://R2/projects/rcmd_feature/hdfs/prod/alg/dd/train_data/graphsage_v0/BR/{day}
    csv_output_path = f"{csv_path}/{region}/{day}"
    # merge_path = get_last_merge_data(merge_base)
    # df = spark.read.parquet(merge_path)
    # df = df.filter(df.w > edge_thre)

    va_hop_path = f"{base_path}/{region}/va_hop/{day}"
    # label_df = spark.read.parquet(day_path).toDF(*['va_day','vb_day','w_day'])
    # va_hop = gen_2hop(label_df, df, spark).toDF(*['tag','va','vb', 'va_hop1','va_hop2'])
    # va_hop.repartition(100).write.format("parquet").mode("overwrite").save(va_hop_path)

    vb_hop_path = f"{base_path}/{region}/vb_hop/{day}"
    # label_df = spark.read.parquet(day_path).toDF(*['vb_day','va_day','w_day'])
    # vb_hop = gen_2hop(label_df, df, spark).toDF(*['tag','vb','va','vb_hop1','vb_hop2'])
    # vb_hop.repartition(100).write.format("parquet").mode("overwrite").save(vb_hop_path)

    va_hop = spark.read.parquet(va_hop_path)
    vb_hop = spark.read.parquet(vb_hop_path)
    res = va_hop.join(vb_hop, (va_hop.tag == vb_hop.tag) & (va_hop.va == vb_hop.va) & (va_hop.vb == vb_hop.vb)) \
        .drop(vb_hop.tag).drop(vb_hop.va).drop(vb_hop.vb)
    # res.repartition(100).write.format("parquet").mode("overwrite").save(output_path)
    csv_res = convert_arr_hops(spark, res)
    csv_res.write.csv(csv_output_path, mode='overwrite', header=False, sep='\t')


def convert_data(spark):
    origin_output_path = f"{base_path}/{region}/train_data/{day}"
    csv_output_path = f"{csv_path}/{region}/{day}"
    hdp_cmd = f"hadoop fs -test -d {csv_output_path}"
    code = subprocess.call(hdp_cmd, shell=True)
    print("CHECKED %s = %s" % (hdp_cmd, code))
    if code != 0:
        res = spark.read.parquet(origin_output_path)
        csv_res = convert_arr_hops(spark, res)
        csv_res.write.csv(csv_output_path, header=False, sep='\t')


if __name__ == "__main__":
    options = usage()
    day = options.day
    region = options.region
    base_path = options.base_path
    csv_path = options.csv_path
    action = options.action
    min_weight_tld = options.min_weight_tld
    merge_days = int(options.merge_days)
    rcmdble_table_path = options.rcmdble_table_path
    edge_thre = int(options.edge_thre)
    hip1_num = int(options.hip1_num)
    hip2_num = int(options.hip2_num)

    ALL_REGIONS = ['SG', 'MY', 'TH', 'TW', 'ID', 'VN', 'PH', 'BR', 'MX', 'PL', 'ES', 'AR', 'CO', 'CL']
    REGIONS_TIMEZONE = {'SG': 8, 'MY': 8, 'TH': 7, 'TW': 8, 'ID': 7, 'VN': 7, 'PH': 8, 'BR': -3, 'MX': -5, 'PL': 1,
                        'ES': 1, 'AR': -3, 'CO': -6, 'CL': -4}

    spark = SparkSession. \
        builder \
        .appName("graphsage/xuehan.tan@shopee.com") \
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

    if action == "gen_data":
        day_data(spark)
        remove_day_data(spark)
    elif options.action == "merge_data":
        merge_result(spark)
        remove_merge_data(spark)
    elif options.action == "gen_va_hop":
        gen_va_hop(spark)
    elif options.action == "gen_vb_hop":
        gen_vb_hop(spark)
    elif options.action == "gen_train_data":
        gen_train_sample(spark)
    elif options.action == "convert_data":
        convert_data(spark)