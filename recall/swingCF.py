#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os, sys

from pyspark.sql import SparkSession, Row
from pyspark.sql.types import DoubleType
from pyspark.sql import functions as F
# from pyspark.types import *
from subprocess import call
from optparse import OptionParser
import datetime as dt
import time
import json
import math
import random
from heapq import merge
from itertools import islice, combinations

ALL_REGIONS = ['SG', 'MY', 'TH', 'TW', 'ID', 'VN', 'PH', 'BR', 'MX']
REGIONS_TIMEZONE = {'SG': 8, 'MY': 8, 'TH': 7, 'TW': 8, 'ID': 7, 'VN': 7, 'PH': 8, 'BR': -3, 'MX': -5}

is_debug = True
## 单次统计共现的时间窗口
window_days = 1
## 每个商品每次统计时最多能够被统计到的用户数
max_item_users = 10000
## 每个商品每次统计时最少能够被统计到的用户数
min_item_users = 5
## 两两用户权重对计算时滑动的时间窗口
window_size = 5000
## 最多保留几次单次计算
day_keep_num = 200
## 最多保留几次合并计算
merge_keep_num = 3
## 最小过滤的阈值权重
min_weight_tld = 3
## 每个user最多保留的相似用户数量
max_user_num_s1 = 1000
## 每个user最多保留的相似用户数量
max_user_num_s2 = 700
## 每个用户最少关联的用户数量
min_user_num = 10
## 用户相似度阈值
min_user_score_tld = 0.01
## 查询多少天用户历史记录
fetch_list_day = 30
## 对每个用户查询时关联的物品最大数量
fetch_max_user_item = 10000
## 对每个用户查询时关联的物品最小数量
fetch_min_user_item = 5
## 对每个用户查询时关联的物品的最终数量的阈值
fetch_final_item_num = 700
## 用户对物品权重阈值
min_item_score_tld = 0.01


def usage():
    parser = OptionParser()
    parser.add_option('--day', dest="day", default="")
    parser.add_option('--hour', dest="hour", default="")
    parser.add_option('--region', dest="region", default="ID")
    parser.add_option('--post_days', dest="post_days", default="30")
    parser.add_option('--merge_days', dest="merge_days", default="90")
    parser.add_option('--action', dest="action", default="gen_data")
    parser.add_option('--partition', dest="partition", default="100")
    parser.add_option('--executors', dest="executors", default="500")
    parser.add_option('--input', dest='input', default='')
    parser.add_option('--output', dest='output', default='')
    parser.add_option('--pos_count_tld', dest='pos_count_tld', default='80')
    parser.add_option('--base_path', dest='base_path',
                      default='your base path')
    parser.add_option('--rcmdble_table_path', dest='rcmdble_table_path',
                      default='your recommendable table path')
    (options, _) = parser.parse_args()
    return options


def extractItemClickData(row):
    ## 权重取负数是为了方便多路归并
    action_weight = 1.0
    return row.itemid, [(-action_weight, row.userid)]


def get_recommendable_items(spark, options):
    region = options.region
    date = options.date

    sql_cmd = '''
        select itemid from recommendable_items where region = '{}' and date= '{}'
        '''.format(region, date)

    if is_debug:
        print("[DEBUG] get_recommendable_items sql is: %s" % sql_cmd)
    df = spark.sql(sql_cmd)
    df.createOrReplaceTempView("rcmd_items")



def generate_u2u_weight(spark, options):
    def extract_item_pair(row):
        userid = row[0]
        item_set = sorted(row[1])
        for itemi, itemj in combinations(item_set, 2):
            yield ((itemi, itemj), userid)

    def extract_user_pair(row):
        user_set = sorted(row[1])
        for ui, uj in combinations(user_set, 2):
            yield ((ui, uj), 1)

    day = options.day

    sql_cmd = '''
        select userid, cast(get_json_object(data, '$.itemid') as BIGINT) as itemid
        FROM {}
        WHERE date='{}' and region='{}'
    '''.format(options.table, options.date, options.region)

    if is_debug:
        print("sql cmd is: %s", sql_cmd)

    user_item_output_path = "%s/%s/user_item_day_data/%s" % (options.base_path, options.region, day)
    cmd_str = "hadoop fs -ls {}/_SUCCESS".format(user_item_output_path)
    cmd_result = os.popen(cmd_str).readlines()
    print(cmd_result)
    flag = "SUCCESS" in cmd_result[0] if len(cmd_result) > 0 else False
    if not flag:
        print("{} does not exists".format(user_item_output_path))
        df = spark.sql(sql_cmd)
        print("size of raw userid_itemid is {}\n".format(df.count()))
        # call("hadoop fs -rmr %s" % user_item_output_path, shell=True)
        df.repartition(500).write.format("parquet").mode("overwrite").save(user_item_output_path)
    else:
        print("{} exists".format(user_item_output_path))

    # generate i2i2U pair
    i2i2u_output_path = "%s/%s/i2i2u_day_data/%s" % (options.base_path, options.region, day)
    cmd_str = "hadoop fs -ls {}/_SUCCESS".format(i2i2u_output_path)
    cmd_result = os.popen(cmd_str).readlines()
    print(cmd_result)
    flag = "SUCCESS" in cmd_result[0] if len(cmd_result) > 0 else False
    if not flag:
        partition = int(options.partition)
        df = spark.sql(sql_cmd)
        df1 = df.groupBy("userid").agg(F.collect_set("itemid").alias("itemid_set")) \
            .rdd.filter(lambda t: len(t[1]) >= 2) \
            .flatMap(extract_item_pair) \
            .toDF()
        print("size of i2i2u pair is {}\n".format(df1.count()))
        df1.repartition(500).write.format("parquet").mode("overwrite").save(i2i2u_output_path)
    else:
        print("{} exists".format(i2i2u_output_path))

    # genarate u2u pair
    u2u_output_path = "%s/%s/u2u_day_data/%s" % (options.base_path, options.region, day)
    cmd_str = "hadoop fs -ls {}/_SUCCESS".format(u2u_output_path)
    cmd_result = os.popen(cmd_str).readlines()
    print(cmd_result)
    flag = "SUCCESS" in cmd_result[0] if len(cmd_result) > 0 else False
    if not flag:
        partition = int(options.partition)
        df = spark.sql(sql_cmd)
        df2 = df.groupBy("itemid").agg(F.collect_set("userid").alias("user_set")) \
            .rdd.filter(lambda t: len(t[1]) >= 2 and len(t[1]) <= 1000) \
            .flatMap(extract_user_pair) \
            .reduceByKey(lambda x, y: x + y) \
            .toDF()  # ((ui, uj), count)
        print("size of u2u pair is {}\n".format(df2.count()))
        df2.repartition(500).write.format("parquet").mode("overwrite").save(u2u_output_path)
    else:
        print("{} exists".format(u2u_output_path))


def remove_day_data(spark, options):
    tags = ["user_item_day_data", "i2i2u_day_data", "u2u_day_data"]
    for tag in tags:
        hour_path = "%s/%s/%s" % (options.base_path, options.region, tag)
        cmd_str = "hadoop fs -ls %s/*/_SUCCESS | awk '{{if(NF==8)print $NF}}' | sort -r" % hour_path
        cmd_res = os.popen(cmd_str).readlines()
        if len(cmd_res) > day_keep_num:
            remove_data = cmd_res[day_keep_num:]
            for filename in remove_data:
                filename = filename.strip('\r\n')
                if filename.endswith('_SUCCESS'):
                    filename = filename[:-len("_SUCCESS")]
                call("hadoop fs -rmr %s" % filename, shell=True)

    hour_path = "%s/%s/user_item_day_data" % (options.base_path, options.region)
    cmd_str = "hadoop fs -ls %s/*/_SUCCESS | awk '{{if(NF==8)print $NF}}' | sort -r" % hour_path
    cmd_res = os.popen(cmd_str).readlines()
    if len(cmd_res) > day_keep_num:
        remove_data = cmd_res[day_keep_num:]
        for filename in remove_data:
            filename = filename.strip('\r\n')
            if filename.endswith('_SUCCESS'):
                filename = filename[:-len("_SUCCESS")]
            call("hadoop fs -rmr %s" % filename, shell=True)


def merge_u_weight_approx1(spark, options):
    merge_days = int(options.merge_days)
    print("merge days: {}".format(merge_days))
    partition = int(options.partition)

    # merge u2u
    u_merge_base = "%s/%s/merge_u_weight_data" % (options.base_path, options.region)
    print(u_merge_base)
    output_path = "%s/%s" % (u_merge_base, options.day)
    cmd_str = "hadoop fs -ls {}/_SUCCESS".format(output_path)
    cmd_result = os.popen(cmd_str).readlines()
    print(cmd_result)
    flag = "SUCCESS" in cmd_result[0] if len(cmd_result) > 0 else False
    if flag:
        print("{} already exists!".format(output_path))
    else:
        files = []
        for i in range(0, merge_days + 1):
            day = (dt.datetime.strptime(options.day, '%Y-%m-%d') - dt.timedelta(days=i)).strftime("%Y-%m-%d")
            filename = "%s/%s/user_item_day_data/%s" % (options.base_path, options.region, day)
            files.append(filename)

        rdds = []
        for filename in files:
            rdd = spark.read.parquet(filename).rdd
            rdds.append(rdd)

        df = spark.sparkContext.union(rdds).map(lambda x: Row(userid=x[0], itemid=x[1])).toDF() \
            .groupBy("userid").agg(F.collect_set("itemid").alias("item_set")) \
            .rdd.map(lambda x: Row(userid=x[0], weight=1.0 / math.sqrt(len(x[1])))).toDF()

        print(df.count())
        df.repartition(512).write.format("parquet").mode("overwrite").save(output_path)


def remove_u_weight_data(spark, options, is_group=False):
    day_path = "%s/%s/merge_u_weight_data" % (options.base_path, options.region)
    cmd_str = "hadoop fs -ls %s/*/_SUCCESS | awk '{{if(NF==8)print $NF}}' | sort -r" % day_path
    cmd_res = os.popen(cmd_str).readlines()
    if len(cmd_res) > merge_keep_num:
        remove_data = cmd_res[merge_keep_num:]
        for filename in remove_data:
            filename = filename.strip('\r\n')
            if filename.endswith("_SUCCESS"):
                filename = filename[:-len("_SUCCESS")]
            call("hadoop fs -rmr %s" % filename, shell=True)


def merge_u2u_weight_approx1(spark, options):
    merge_days = int(options.merge_days)
    print("merge days: {}".format(merge_days))
    partition = int(options.partition)

    # merge u2u
    u2u_merge_base = "%s/%s/merge_u2u_data" % (options.base_path, options.region)
    print(u2u_merge_base)
    output_path = "%s/%s" % (u2u_merge_base, options.day)
    cmd_str = "hadoop fs -ls {}/_SUCCESS".format(output_path)
    cmd_result = os.popen(cmd_str).readlines()
    print(cmd_result)
    flag = "SUCCESS" in cmd_result[0] if len(cmd_result) > 0 else False
    if flag:
        print("{} already exists!".format(output_path))
    else:
        files = []
        for i in range(0, merge_days + 1):
            day = (dt.datetime.strptime(options.day, '%Y-%m-%d') - dt.timedelta(days=i)).strftime("%Y-%m-%d")
            filename = "%s/%s/u2u_day_data/%s" % (options.base_path, options.region, day)
            files.append(filename)

        rdds = []
        for filename in files:
            rdd = spark.read.parquet(filename).rdd
            rdds.append(rdd)

        df = spark.sparkContext.union(rdds).reduceByKey(lambda x, y: x + y, 512) \
            .map(lambda t: Row(ui=t[0][0], uj=t[0][1], weight=t[1])) \
            .toDF()
        print(df.count())
        df.repartition(512).write.format("parquet").mode("overwrite").save(output_path)


def remove_u2u_data(spark, options, is_group=False):
    day_path = "%s/%s/merge_u2u_data" % (options.base_path, options.region)
    cmd_str = "hadoop fs -ls %s/*/_SUCCESS | awk '{{if(NF==8)print $NF}}' | sort -r" % day_path
    cmd_res = os.popen(cmd_str).readlines()
    if len(cmd_res) > merge_keep_num:
        remove_data = cmd_res[merge_keep_num:]
        for filename in remove_data:
            filename = filename.strip('\r\n')
            if filename.endswith("_SUCCESS"):
                filename = filename[:-len("_SUCCESS")]
            call("hadoop fs -rmr %s" % filename, shell=True)


def merge_i2i2u2u_weight_approx1(spark, options):
    def extract_user_pair(row):
        itemi_itemj = row[0]
        user_set = sorted(row[1])
        for ui, uj in combinations(user_set, 2):
            yield (itemi_itemj, ui, uj)

    merge_days = int(options.merge_days)
    print("merge days: {}".format(merge_days))
    # merge i2i2u2u
    i2i2u2u_merge_base = "%s/%s/merge_i2i2u2u_data" % (options.base_path, options.region)
    print(i2i2u2u_merge_base)
    output_path = "%s/%s" % (i2i2u2u_merge_base, options.day)
    cmd_str = "hadoop fs -ls {}/_SUCCESS".format(output_path)
    cmd_result = os.popen(cmd_str).readlines()
    print(cmd_result)
    flag = "SUCCESS" in cmd_result[0] if len(cmd_result) > 0 else False
    if flag:
        print("{} already exists!".format(output_path))
    else:
        files = []
        for i in range(0, merge_days + 1):
            day = (dt.datetime.strptime(options.day, '%Y-%m-%d') - dt.timedelta(days=i)).strftime("%Y-%m-%d")
            filename = "%s/%s/i2i2u_day_data/%s" % (options.base_path, options.region, day)
            files.append(filename)

        rdds = []
        for filename in files:
            rdd = spark.read.parquet(filename).rdd
            rdds.append(rdd)

        df = spark.sparkContext.union(rdds).map(lambda x: Row(itemi_itemj=x[0], userid=x[1])).toDF() \
            .groupBy("itemi_itemj").agg(F.collect_set("userid").alias("userid_set")) \
            .rdd.repartition(512).flatMap(extract_user_pair) \
            .map(lambda t: Row(itemi=t[0][0], itemj=t[0][1], ui=t[1], uj=t[2])) \
            .toDF()

        print(df.count())
        output_path = "%s/%s" % (i2i2u2u_merge_base, options.day)
        df.repartition(512).write.format("parquet").mode("overwrite").save(output_path)


def remove_i2i2u2u_data(spark, options, is_group=False):
    day_path = "%s/%s/merge_i2i2u2u_data" % (options.base_path, options.region)
    cmd_str = "hadoop fs -ls %s/*/_SUCCESS | awk '{{if(NF==8)print $NF}}' | sort -r" % day_path
    cmd_res = os.popen(cmd_str).readlines()
    if len(cmd_res) > merge_keep_num:
        remove_data = cmd_res[merge_keep_num:]
        for filename in remove_data:
            filename = filename.strip('\r\n')
            if filename.endswith("_SUCCESS"):
                filename = filename[:-len("_SUCCESS")]
            call("hadoop fs -rmr %s" % filename, shell=True)


def remove_join_data(spark, options, is_group=False):
    day_path = "%s/%s/join_data" % (options.base_path, options.region)
    cmd_str = "hadoop fs -ls %s/*/_SUCCESS | awk '{{if(NF==8)print $NF}}' | sort -r" % day_path
    cmd_res = os.popen(cmd_str).readlines()
    if len(cmd_res) > merge_keep_num:
        remove_data = cmd_res[merge_keep_num:]
        for filename in remove_data:
            filename = filename.strip('\r\n')
            if filename.endswith("_SUCCESS"):
                filename = filename[:-len("_SUCCESS")]
            call("hadoop fs -rmr %s" % filename, shell=True)


def remove_group_data(spark, options, is_group=False):
    day_path = "%s/%s/final_data" % (options.base_path, options.region)
    cmd_str = "hadoop fs -ls %s/*/_SUCCESS | awk '{{if(NF==8)print $NF}}' | sort -r" % day_path
    cmd_res = os.popen(cmd_str).readlines()
    if len(cmd_res) > merge_keep_num:
        remove_data = cmd_res[merge_keep_num:]
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


def join_data(spark, options):
    get_recommendable_items(spark, options)
    print("got recommendable items")

    i2i2u2u_merge_base = '%s/%s/merge_i2i2u2u_data' % (options.base_path, options.region)
    merge_path = get_last_merge_data(i2i2u2u_merge_base)
    if merge_path == None:
        print("merge data is not ready, path: %s" % merge_path)
        sys.exit(-1)
    if is_debug:
        print("[DEBUG] merge_path: %s" % merge_path)
    df_i2i2u2u = spark.read.parquet(merge_path)
    df_i2i2u2u.createOrReplaceTempView("itemi_itemj_ui_uj_raw")
    print("itemi_itemj_ui_uj_raw")

    u_merge_base = '%s/%s/merge_u_weight_data' % (options.base_path, options.region)
    merge_path = get_last_merge_data(u_merge_base)
    if merge_path == None:
        print("merge data is not ready, path: %s" % merge_path)
        sys.exit(-1)
    if is_debug:
        print("[DEBUG] merge_path: %s" % merge_path)
    df_u_weight = spark.read.parquet(merge_path)
    df_u_weight.createOrReplaceTempView("u_weight")
    print("u_weight")

    u2u_merge_base = '%s/%s/merge_u2u_data' % (options.base_path, options.region)
    merge_path = get_last_merge_data(u2u_merge_base)
    if merge_path == None:
        print("merge data is not ready, path: %s" % merge_path)
        sys.exit(-1)
    if is_debug:
        print("[DEBUG] merge_path: %s" % merge_path)
    df_u2u = spark.read.parquet(merge_path)
    df_u2u.createOrReplaceTempView("ui_uj_weight")
    print("ui_uj_weight")

    spark.sql("""
        create or replace temporary view itemi_itemj_ui_uj as
        select a.itemi, a.itemj, a.ui, a.uj
        from itemi_itemj_ui_uj_raw a 
        join rcmd_items b on a.itemi=b.itemid
        join rcmd_items c on a.itemj=c.itemid   
    """)

    spark.sql("""
        create or replace temporary view ui_uj_uiw_ujw_weight as
        select a.ui, a.uj, b.weight as uiw, c.weight as ujw, a.weight as weight
        from 
        ui_uj_weight as a
        join u_weight as b on a.ui = b.userid
        join u_weight as c on a.uj = c.userid
    """)

    spark.sql("""
        create or replace temporary view itemi_itemj_score as
        select itemi, itemj, sum(score) as swing_score from 
        (
        select a.itemi, a.itemj, uiw * ujw * 1/(1 + b.weight) as score from 
        (select itemi, itemj, ui, uj from itemi_itemj_ui_uj) as a
        join 
        (select ui, uj, uiw, ujw, weight from ui_uj_uiw_ujw_weight) as b
        on a.ui = b.ui and a.uj = b.uj
        )
        group by itemi, itemj
    """)
    print("itemi_itemj_score")

    spark.sql("""
        create or replace temporary view itemj_itemi_score as
        select itemj as itemi, itemi as itemj, swing_score from itemi_itemj_score
        """)
    print("itemj_itemi_score")

    df = spark.sql("""
        select * from itemi_itemj_score
        union 
        select * from itemj_itemi_score
        """)  # .rdd.repartition(512).map(lambda t: Row(itemi=t[0], itemj=t[1], swing_score=t[2])).toDF()
    print("count of itemi_itemj_score: {}".format(df.count()))

    output_path = "%s/%s/join_data" % (options.base_path, options.region)
    call("hadoop fs -rmr %s" % output_path, shell=True)
    df.repartition(512).write.format("parquet").mode("overwrite").save(output_path)

    print("start generating case examples...")
    case_debug_path = "%s/%s/case_data" % (options.base_path, options.region)
    call("hadoop fs -rmr %s" % case_debug_path, shell=True)
    df.createOrReplaceTempView("itemi_itemj_swing")

    spark.sql("""
    create or replace temporary view item_shop as
    select item_id, shop_id from shopee.order_mart_dws_item_gmv_nd where tz_type='local' and grass_region='ID' and grass_date=date'{}'
    """.format(options.day))

    df = spark.sql("""
    select a.itemi, concat_ws(':', b.shop_id, a.itemi) as shopi_itemi, a.itemj, concat_ws(':', c.shop_id, a.itemj) as shopj_itemj, swing_score
    from itemi_itemj_swing a
    join item_shop b on a.itemi = b.item_id
    join item_shop c on a.itemj = c.item_id
    """).groupBy("shopi_itemi").agg(F.collect_list(F.col("shopj_itemj"))).rdd.map(
        lambda x: x[0] + " " + " ".join(x[1][:100]))
    df.repartition(512).saveAsTextFile(case_debug_path, "org.apache.hadoop.io.compress.GzipCodec")


def group_data(spark, options):
    def show_pencentile(path):
        sc = spark.sparkContext
        final_result = sc.textFile(path)
        final_result.map(lambda line: len(line.split("\t")[1].split(";"))).map(
            Row("length")).toDF().createOrReplaceTempView("size")
        df = spark.sql("select count(length) as key_size from size")
        df.show()
        df = spark.sql("""
        select avg(length) as avg_length,  percentile(length, 0.1) as per10, percentile(length, 0.2) as per20, percentile(length, 0.3) as per30, percentile(length, 0.4) as per40, percentile(length, 0.5) as per50
        , percentile(length, 0.6) as per60, percentile(length, 0.7) as per70, percentile(length, 0.8) as per80, percentile(length, 0.9) as per90 from size
        """)
        df.show()

    row = Row("result")  # Or some other column name
    input_path = "%s/%s/join_data" % (options.base_path, options.region)
    df_raw = spark.read.parquet(input_path)
    df = df_raw.groupBy("itemi").agg(
        F.collect_list(F.concat(F.col('itemj'), F.lit(':'), F.col('swing_score')))).rdd.map(
        lambda x: "%s\t%s" % (x[0], ";".join(sorted(x[1], key=lambda y: float(y.split(":")[1]), reverse=True)[:200])))
    print("count of final result: {}".format(df.map(row).toDF().count()))
    output_path = "%s/%s/final_data" % (options.base_path, options.region)
    call("hadoop fs -rmr %s" % output_path, shell=True)
    df.repartition(512).saveAsTextFile(output_path, "org.apache.hadoop.io.compress.GzipCodec")
    show_pencentile(output_path)

    df_raw.createOrReplaceTempView("itemi_itemj_swing")

    df = spark.sql("""
    select cast(max(dt) as string) as date from rcmd_feature.dws_item_feature where country='{}'
    """.format(options.region))
    latest_day = df.select("date").rdd.collect()[0].date
    print("latest partition day of rcmd_feature.dws_item_feature is {}".format(latest_day))

    spark.sql("""
    create or replace temporary view item_ctr_cvr as 
    select itemid, ctcvr30
    from rcmd_feature.dws_item_feature
    where country = '{}' and dt = '{}'
    """.format(options.region, latest_day))
    df_ctr_cr = spark.sql("""
    select a.itemi, a.itemj, ctcvr30 * a.swing_score as swing_score
    from itemi_itemj_swing as a
    join item_ctr_cvr as b on a.itemj = b.itemid
    """)
    df = df_ctr_cr.groupBy("itemi").agg(
        F.collect_list(F.concat(F.col('itemj'), F.lit(':'), F.col('swing_score')))).rdd.map(
        lambda x: "%s\t%s" % (x[0], ";".join(sorted(x[1], key=lambda y: float(y.split(":")[1]), reverse=True)[:200])))
    print("count of final result: {}".format(df.map(row).toDF().count()))
    output_path = "%s/%s/final_data_ctr_cr" % (options.base_path, options.region)
    call("hadoop fs -rmr %s" % output_path, shell=True)
    df.repartition(512).saveAsTextFile(output_path, "org.apache.hadoop.io.compress.GzipCodec")
    show_pencentile(output_path)


def fetch_user_click_list(spark, options):
    def extract_user_item(row):
        detail_obj = json.loads(row.data)
        str_itemid = str(detail_obj.get("itemid", -1))
        itemid = 0
        if str_itemid.isdigit():
            itemid = int(str_itemid)
        action_weight = 1.0
        return row.userid, [(-row.event_timestamp, itemid, action_weight)]

    def sort_and_truncate(line):
        _, values = line
        return ','.join(
            [str(values[i][1]) + ":" + str(values[i][2]) for i in range(min(fetch_final_item_num, len(values)))])

    def group_map_score(row):
        score = float(row.weight) / math.sqrt(row.weight_a) / math.sqrt(row.weight_b)
        yield (row.ua, row.ub, score)
        yield (row.ub, row.ua, score)

    def extract_item_score(row):
        u1, score, itemlist = row.u1, row.score, row.itemlist
        item_score_list = [(v.split(':')[0], float(v.split(':')[1])) for v in itemlist.strip('\n').split(',')]
        score_map = {}
        for item_score in item_score_list:
            item, value = item_score[0], item_score[1]
            score_map[item] = score_map.get(item, 0) + value
        for key, value in score_map.items():
            yield (u1, key), score * value

    def map_key_value(line):
        (user, item), value = line
        return user, [(-value, item)]

    def truncat_final_itemlist(line):
        _, item_score_list = line
        return ','.join(c[1] for c in item_score_list)

    day = options.day
    dep_start_date = dt.datetime.strptime(day, '%Y-%m-%d') - dt.timedelta(days=fetch_list_day)
    dep_end_date = dt.datetime.strptime(day, '%Y-%m-%d') - dt.timedelta(days=0)
    dep_start_date_str = dep_start_date.strftime('%Y-%m-%d')
    dep_end_date_str = dep_end_date.strftime('%Y-%m-%d')
    end_hour = 23 - REGIONS_TIMEZONE[options.region]
    start_hour = end_hour + 1

    b_itemset = get_recommendable_items(spark, options)
    sql_cmd = '''
        select userid, event_timestamp, operation, data FROM shopee.traffic_mart_dwd__click_di
        where page_type='home' and page_section[0] = 'daily_discover' 
        and data is not null and length(data) > 0 
        and target_type='item' and operation = 'click' and
        ((utc_date=date'{}' and hour <= {}) or (utc_date = DATE'{}' and hour >= {})) and userid > 0 and grass_region='{}'
    '''.format(dep_end_date_str, end_hour, dep_start_date_str, start_hour, options.region)
    df = spark.sql(sql_cmd)
    partition = int(options.partition)
    df = df.rdd.repartition(partition).map(extract_user_item) \
        .filter(lambda t: (t[1][0][1] != 0) and (t[1][0][1] in b_itemset.value)) \
        .reduceByKey(lambda x, y: list(islice(merge(x, y), fetch_final_item_num)), partition) \
        .map(lambda t: Row(u=t[0], itemlist=sort_and_truncate(t))) \
        .toDF()

    if is_debug:
        print("user_to_itemlist first 10 rows\n")
        print(df.head(10))
    df.createOrReplaceTempView("user_to_itemlist_table")

    input_path = "%s/%s/join_cnt" % (options.base_path, options.region)
    u2u_df = spark.read.parquet(input_path).rdd \
        .flatMap(group_map_score) \
        .repartition(int(options.partition)) \
        .map(lambda t: Row(u1=t[0], u2=t[1], score=t[2])) \
        .toDF()

    if is_debug:
        print("user_to_user first 10 rows\n")
        print(u2u_df.head(10))
    u2u_df.createOrReplaceTempView("user_to_user_table")

    sql_cmd = '''
    select t1.u1 as u1, t1.score as score, t2.itemlist as itemlist from user_to_user_table t1 inner join user_to_itemlist_table t2 on t1.u2 = t2.u
    '''
    combined_df = spark.sql(sql_cmd)

    if is_debug:
        print("combined_df first 10 row\n")
        print(combined_df.head(10))
    rdd = combined_df.rdd.repartition(partition) \
        .flatMap(extract_item_score) \
        .reduceByKey(lambda x, y: x + y, partition) \
        .map(map_key_value) \
        .reduceByKey(lambda x, y: list(islice(merge(x, y), fetch_final_item_num)), partition) \
        .map(lambda t: "%s\t%s" % (t[0], truncat_final_itemlist(t)))

    output_path = "%s/%s/recommended_list_test" % (options.base_path, options.region)
    call("hadoop fs -rmr %s" % output_path, shell=True)
    rdd.repartition(100).saveAsTextFile(output_path, "org.apache.hadoop.io.compress.GzipCodec")


def offline_evalution(spark, options):
    def extractEvalutionItemClickData(row):
        detail_obj = json.loads(row.data)
        str_itemid = str(detail_obj.get("itemid", -1))
        itemid = 0
        if str_itemid.isdigit():
            itemid = int(str_itemid)
        return row.userid, [(itemid, row.event_timestamp)]

    def extractList(line):
        userid, itemlist = line
        itemlist.sort(key=lambda t: t[1], reverse=True)
        return userid, ','.join(str(c[0]) for c in itemlist)

    def hit_ratio(label_list, predict_list):
        label_list = label_list.split(',')
        label_list = set(label_list)
        predict_list = predict_list.split(',')
        count = 0
        for predict_item in predict_list:
            if predict_item in label_list:
                count += 1
        return count * 1.0 / len(label_list)

    day = options.day
    dep_start_date = dt.datetime.strptime(day, '%Y-%m-%d')
    dep_end_date = dt.datetime.strptime(day, '%Y-%m-%d') + dt.timedelta(days=1)
    dep_start_date_str = dep_start_date.strftime('%Y-%m-%d')
    dep_end_date_str = dep_end_date.strftime('%Y-%m-%d')
    end_hour = 23 - REGIONS_TIMEZONE[options.region]
    start_hour = end_hour + 1

    sql_cmd = '''
        select userid, event_timestamp, operation, data FROM shopee.traffic_mart_dwd__click_di
        where page_type='home' and page_section[0] = 'daily_discover' 
        and data is not null and length(data) > 0 
        and target_type='item' and operation = 'click' and
        ((utc_date=date'{}' and hour <= {}) or (utc_date = DATE'{}' and hour >= {})) and userid > 0 and grass_region='{}'
    '''.format(dep_end_date_str, end_hour, dep_start_date_str, start_hour, options.region)
    df = spark.sql(sql_cmd)
    partition = int(options.partition)
    # if is_debug:
    #     print("[eval]: user2itemlist first 10 rows: ")
    #     print(df.head(10))

    df = df.rdd.repartition(partition).map(extractEvalutionItemClickData) \
        .filter(lambda t: t[1][0] != 0) \
        .reduceByKey(lambda x, y: list(merge(x, y)), partition) \
        .map(extractList) \
        .map(lambda t: Row(userid=t[0], labellist=t[1])) \
        .toDF()

    df.createOrReplaceTempView("label_table")

    if is_debug:
        print("label_table first 10 rows\n")
        print(df.head(10))

    predict_input_path = "%s/%s/recommended_list_test" % (options.base_path, options.region)

    predict_df = spark.sparkContext.textFile(predict_input_path) \
        .map(lambda t: Row(userid=t.split('\t')[0], predictlist=t.split('\t')[1])) \
        .toDF()
    predict_df.createOrReplaceTempView("predict_table")

    if is_debug:
        print("predict_table first 10 rows\n")
        print(predict_df.head(10))

    spark.udf.register("hit_ratio_score", hit_ratio, returnType=DoubleType())

    sql_cmd = '''
        select t1.userid, hit_ratio_score(t1.labellist, t2.predictlist) as score from label_table t1 inner join predict_table t2 on t1.userid = t2.userid
    '''
    res_df = spark.sql(sql_cmd)
    output_path = "%s/%s/eval_result" % (options.base_path, options.region)
    call("hadoop fs -rmr %s" % output_path, shell=True)
    res_df.rdd.repartition(100).saveAsTextFile(output_path, "org.apache.hadoop.io.compress.GzipCodec")


def main():
    options = usage()
    # start of main program
    spark = SparkSession. \
        builder \
        .appName("ucf_dd_v1/zhiyuan.xu@shopee.com") \
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
    # .config("spark.eventLog.dir", "hdfs://R2/projects/%s/hdfs/dev/sparklogs" % PROJECT_NAME)\
    spark.sparkContext.setLogLevel('ERROR')

    if options.action == "gen_data":
        # 当天数据，产生((itemi, itemj), userid)和((ui, uj), count)
        generate_u2u_weight(spark, options)
        remove_day_data(spark, options)
    elif options.action == "merge_u2u":
        # 多天数据reduce得到((ui, uj), count)
        merge_u2u_weight_approx1(spark, options)
        remove_u2u_data(spark, options)
    elif options.action == "merge_u_weight":
        # 得到swing公式里面user weight
        merge_u_weight_approx1(spark, options)
        remove_u_weight_data(spark, options)
    elif options.action == "merge_i2i2u2u":
        # 得到(itemi,itemj,u1,u2)
        merge_i2i2u2u_weight_approx1(spark, options)
        remove_i2i2u2u_data(spark, options)
    elif options.action == "join_result":
        join_data(spark, options)
        remove_join_data(spark, options)
    elif options.action == "group_result":
        group_data(spark, options)
        remove_group_data(spark, options)
    # elif options.action == "generate_user_item_list":
    #    fetch_user_click_list(spark, options)
    # elif options.action == "offline_evalution":
    #    offline_evalution(spark, options)


if __name__ == '__main__':
    main()