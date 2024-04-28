import pandas as pd
import datetime


def intermediate_partition(frame: pd.DataFrame, categories=None) -> pd.DataFrame:
    """
    Function to aggregate dataframe to 6 interval categories for each weekday

    :param frame: Initial frame
    :param categories: Mapping dictionary
    :return: Dataframe with open, high, low, close, volume, date, weekday and interval
    """

    if categories is None:
        categories = {'0-4': 0, '4-8': 1, '8-12': 2, '12-16': 3, '16-20': 4, '20-24': 5}

    tmp_data_ = frame.copy()
    tmp_data_['weekday'] = tmp_data_['open_time'].dt.day_name()
    tmp_data_['date'] = tmp_data_['open_time'].dt.date
    tmp_data_['open_time_tmp'] = tmp_data_['open_time'].dt.time
    tmp_data_['close_time_tmp'] = tmp_data_['close_time'].dt.time

    tmp_list_ = []

    for weekday in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        day_frame_ = tmp_data_.loc[tmp_data_['weekday'] == weekday]
        for date in day_frame_['date'].unique():
            date_frame_ = day_frame_.loc[day_frame_['date'] == date]
            for index in [4, 8, 12, 16, 20, 24]:
                start_ = index - 4
                stop_ = index

                if index == 24:
                    interval_frame_ = date_frame_.loc[date_frame_['open_time_tmp'] >= datetime.time(start_)]
                else:
                    interval_frame_ = date_frame_.loc[(date_frame_['open_time_tmp'] >= datetime.time(start_)) & (
                            date_frame_['close_time_tmp'] < datetime.time(stop_))]

                tmp_list_.append([interval_frame_['open'][interval_frame_.index[0]],
                                  interval_frame_['close'][interval_frame_.index[-1]],
                                  interval_frame_['high'].max(),
                                  interval_frame_['low'].min(),
                                  interval_frame_['volume'].sum(),
                                  interval_frame_['n_trades'].sum(),
                                  date,
                                  weekday,
                                  f"{start_}-{stop_}"])

    columns_ = ["open", "close", "high", "low", "volume", "n_trades", "date", "weekday", "interval"]

    final_frame_ = pd.DataFrame(tmp_list_, columns=columns_)
    final_frame_['interval'] = pd.Categorical(final_frame_['interval'],
                                              categories=categories, ordered=True)
    final_frame_.sort_values(by=['date', 'interval'], inplace=True)
    final_frame_.reset_index(drop=True, inplace=True)

    return final_frame_


def partition(frame: pd.DataFrame) -> dict:
    inter_part_ = intermediate_partition(frame)

    tmp_dict_ = {}
    for weekday in inter_part_['weekday'].unique():
        tmp_dict_[weekday] = {}
        for interval in inter_part_['interval'].unique():
            tmp_part_ = inter_part_.loc[(inter_part_['weekday'] == weekday) & (inter_part_['interval'] == interval)]
            tmp_part_ = tmp_part_.drop(columns=['weekday', 'interval'])

            tmp_dict_[weekday][interval] = tmp_part_[['date', 'open', 'high', 'low', 'close', 'volume', 'n_trades']]

    sorted_dict = {day: tmp_dict_[day] for day in
                   ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']}

    return sorted_dict


def prepare_data_for_ranking(data_, variable: str) -> dict:
    tmp_dict_ = {}
    for i in ['0-4', '4-8', '8-12', '12-16', '16-20', '20-24']:
        tmp_interval_ = [data_[weekday][i][variable].values for weekday in data_.keys()]
        tmp_interval_ = pd.DataFrame(tmp_interval_).transpose()
        tmp_interval_.columns = data_.keys()
        tmp_dict_[i] = tmp_interval_

    return tmp_dict_


def prepare_ranking(data: dict) -> dict:
    rankings = {}
    for interval in data.keys():
        ls = []
        for series in data[interval].values:
            try:
                sorted_values = sorted(series, reverse=True)
                map_ = {key: value for key, value in zip(sorted_values, range(1, len(sorted_values) + 1))}
                ls.append([map_[s] for s in series])
            except:
                pass

        ranking = pd.DataFrame(ls,
                               columns=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

        tmp_ls = []
        for w in ranking.columns:
            tmp_ls.append({a: b for a, b in ranking[w].value_counts().items()})

        for index, value in enumerate(tmp_ls):
            tmp_ls[index] = dict(sorted(value.items()))

        ranking_frame = pd.DataFrame(tmp_ls).transpose()
        ranking_frame.columns = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        rankings[interval] = ranking_frame

    return rankings
