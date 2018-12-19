# -*- coding: utf-8 -*-
"""
Created on  2017/9/18 14:46
@author: yulijun
@desc:
"""
import pandas as pd
import numpy as np
import math
import os

class woe_iv(object):
    def __init__(self):
        self.VERSION = 0.1

    @classmethod
    def WOE(cls, data, varList, type0='Con', target_id='y', resfile='result.xlsx'):
        '''
        # 对分类变量直接进行分组统计并进行WOE、IV值 计算
        # 对连续型变量进行分组（default:10）后进行WOE、IV值 计算
        # 最终返回分组统计结果DataFrame格式，并生成xlsx文件
        # 返回字段：'features', 'IV', 'ratio', 'bin_min', 'bin_max', 'woe', 'event_sample', 'nonevent_sample', '占比'
        '''
        result = pd.DataFrame()
        for var in varList:
            print(var)
            try:
                if type0.upper() == "CON".upper():
                    df, retbins = pd.qcut(data[var], q=10, retbins=True, duplicates="drop")
                    tmp = pd.crosstab(df, data[target_id], margins=True)
                    tmp2 = pd.crosstab(df, data[target_id], margins=True).apply(lambda x: x / float(x[-1]), axis=1)
                else:
                    df = data[var]
                    tmp = pd.crosstab(data[var], data[target_id], margins=True)
                    tmp2 = pd.crosstab(data[var], data[target_id], margins=True).apply(lambda x: x / float(x[-1]),
                                                                                       axis=1)
                res = tmp.merge(tmp2, how="left", left_index=True, right_index=True)
                res['ratio'] = res['All_x'] / res['All_x'][-1] * 100
                res['DB'] = (res['0_x']) / res['0_x'][-1]  # Adjusting Woe +0.5
                res['DG'] = (res['1_x']) / res['1_x'][-1]  # Adjusting Woe +0.5
                res['WOE'] = np.log(res['DG'] / res['DB'])
                res['DG-DB'] = res['DG'] - res['DB']
                res['IV'] = res['WOE'] * res['DG-DB']
                res['name'] = var
                res.index.name = ""
                res = res.drop("All")
                res['IV_sum'] = res['IV'].sum()
                del res['0_y']
                del res['All_y']
                if type0.upper() == "CON".upper():
                    res['low'] = retbins[:-1]
                    res['high'] = retbins[1:]
                    res.index = range(len(retbins) - 1)
                else:
                    res['low'] = res.index
                    res['high'] = res.index
                    res.reset_index
                res = res[
                    ['name', 'All_x', 'ratio', 'low', 'high', '0_x', '1_x', '1_y', 'DB', 'DG', 'WOE',
                     'DG-DB',
                     'IV', 'IV_sum']]
                result = result.append(res)

            except Exception as e:
                print(e, var)
        result.to_excel(resfile)
        return result

    @classmethod
    def applyWOE(cls, X_data, X_map, var_list, id_cols_list=None, flag_y=None):
        """
        将最优分箱的结果WOE值对原始数据进行编码
        ------------------------------------------------
        Params
        X_data: pandas DataFrame
        bin_map: pandas dataframe, map table, result of applyhing WOE
        var_list: variable list
        flag_y: y flag when gen the train data
        ------------------------------------------------
        Return
        bin_df: pandas DataFrame, result of bining with y flag
        """
        if flag_y:
            bin_df = X_data[[flag_y]]
        else:
            bin_df = pd.DataFrame(X_data.index)
        for var in var_list:
            x = X_data[var]
            bin_map = X_map[X_map['name'] == var]
            bin_res = np.array([0] * x.shape[-1], dtype=float)
            for i in bin_map.index:
                upper = bin_map['high'][i]
                lower = bin_map['low'][i]
                if lower == upper:
                    # print var,'==============',lower
                    x1 = x[np.where(x == lower)[0]]
                else:
                    # print var, '<<<<<<<<<<<<<<<<<<<<',lower, upper
                    if i == bin_map.index.min():
                        x1 = x[np.where((x <= upper))[0]]  # 会去筛选矩阵里面符合条件的值
                    elif i == bin_map.index.max():
                        x1 = x[np.where((x > lower))[0]]  # 会去筛选矩阵里面符合条件的值
                    else:
                        x1 = x[np.where((x > lower) & (x <= upper))[0]]  # 会去筛选矩阵里面符合条件的值
                #mask = np.in1d(x, x1)  # 用于测试一个数组中的值在另一个数组中的成员资格,返回布尔型数组
                mask = np.in1d(x, x1)  # 用于测试一个数组中的值在另一个数组中的成员资格,返回布尔型数组
                bin_res[mask] = bin_map['WOE'][i]  # 将Ture的数据替换掉
            bin_res = pd.Series(bin_res, index=x.index)
            bin_res.name = x.name
            bin_df = pd.merge(bin_df, pd.DataFrame(bin_res), left_index=True, right_index=True)
        if id_cols_list:
            bin_df = pd.merge(bin_df, X_data[id_cols_list], left_index=True, right_index=True)
        return bin_df

if __name__ == '__main__':

    # generate features data
    data = pd.DataFrame(np.random.randn(100, 4), columns=['x1', 'x2', 'x3', 'x4'])
    data['x5'] = data['x4'].apply(lambda x: ['3', 'Null'][x > 0])
    data['x6'] = data['x4'].apply(lambda x: np.random.choice(['a','b', 'c']))

    # generate target lable
    data['y'] = np.random.randint(0, 2, 100)
    flag_cols = 'y'

    # con example
    cols1 = ['x4', 'x2']
    con_woe = woe_iv.WOE(data, cols1, 'con',flag_cols, 'con.xlsx')
    print(con_woe)
    d = woe_iv.applyWOE(data, con_woe, cols1, False,flag_cols)
    print(d.head())

    # cat example
    cols = ['x5', 'x6']
    cat_woe = woe_iv.WOE(data, cols, 'cat',flag_cols, 'cat.xlsx')
    print(con_woe)
    d = woe_iv.applyWOE(data, cat_woe, cols,cols1,flag_cols )
    print(d.head())

