import torch
from torch.utils.data import Dataset

import calendar
from datetime import datetime, timedelta

class FlightDataset(Dataset):
    def __init__(self,
                 df,
                 route_dict,
                 aircraft_type_dict,
                 aircraft_name_dict,
                 max_min_seating_capacity_dict,
                 max_min_baggage_weight_dict):
        self.df = df
        self.route_dict = route_dict
        self.aircraft_type_dict = aircraft_type_dict
        self.aircraft_name_dict = aircraft_name_dict
        self.max_min_seating_capacity_dict = max_min_seating_capacity_dict
        self.max_min_baggage_weight_dict = max_min_baggage_weight_dict

    def transform(self, row):
        aircraft_name = self.aircraft_name_dict[row['ACFT_REGN']]
        aircraft_name_rep = torch.zeros(self.aircraft_name_dict.__len__())
        aircraft_name_rep[aircraft_name] = 1

        aircraft_type = self.aircraft_type_dict[row['ACFT_Type']]
        aircraft_type_rep = torch.zeros(self.aircraft_type_dict.__len__())
        aircraft_type_rep[aircraft_type] = 1

        route = self.route_dict[row['Route']]
        route_rep = torch.zeros(self.route_dict.__len__())
        route_rep[route] = 1

        weekday_rep = torch.zeros(7)
        date = datetime.strptime(row['DATOP_UTC'], '%d/%m/%Y').date()
        weekday_rep[calendar.weekday(year=date.year, month=date.month, day=date.day)] = 1

        schedule_departure_time = datetime.strptime(row['STD_UTC'].split('T')[1], '%H:%M:%S').time()
        schedule_departure_time_rep = self.trans_time(schedule_departure_time)

        schedule_arrival_time = datetime.strptime(row['STD_UTC'].split('T')[1], '%H:%M:%S').time()
        schedule_arrival_time_rep = self.trans_time(schedule_arrival_time)

        pax_adult_rep = torch.tensor([row['PAX_Adult']/row['Seating_Capacity']])
        pax_inf_rep = torch.tensor([row['PAX_Inf']/10])

        seating_capacity_min_diff = row['Seating_Capacity'] - self.max_min_seating_capacity_dict['min']
        seating_capacity_max_min_diff = self.max_min_seating_capacity_dict['max'] - self.max_min_seating_capacity_dict['min']
        scaled_seating_capacity = seating_capacity_min_diff / seating_capacity_max_min_diff
        seating_capacity_rep = torch.tensor([scaled_seating_capacity])

        baggage_weight_min_diff = row['BaggageWeight_KG'] - self.max_min_baggage_weight_dict['min']
        baggage_weight_max_min_diff = self.max_min_baggage_weight_dict['max'] - self.max_min_baggage_weight_dict['min']
        scaled_baggage_weight = baggage_weight_min_diff / baggage_weight_max_min_diff
        baggage_weight_rep = torch.tensor([scaled_baggage_weight])

        row_rep = torch.cat(
            [aircraft_name_rep.float(),
             aircraft_type_rep.float(),
             route_rep.float(),
             weekday_rep.float(),
             schedule_departure_time_rep.float(),
             schedule_arrival_time_rep.float(),
             pax_adult_rep.float(),
             pax_inf_rep.float(),
             seating_capacity_rep.float(),
             baggage_weight_rep].float(),
            dim=0
        )

        return row_rep

    def trans_time(self, time):
        time_ = timedelta(hours=time.hour, minutes=time.minute, seconds=time.second)
        time_length = time_.total_seconds()

        return torch.tensor([time_length/(24*60*60)])

    def input_dim(self):
        return self.transform(self.df.iloc[0]).shape[0]

    def get_label(self, row):
        return 1 if (row['ARR_Delay_MINS'] + row['DEP_Delay_MINS']) > 0 else 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self,index):
        row = self.df.iloc[index]
        row_feature = self.transform(row)
        label = self.get_label(row)

        return {'rep':row_feature, 'label':label}

class InfoPrefix():
    def __init__(self, flight_df):
        self.flight_schedule_df = flight_df

    def extract_route(self):
        return {route:id for id, route in enumerate(set(self.flight_schedule_df["Route"]))}

    def extract_aircraft_type(self):
        return {aircraft_type:id for id, aircraft_type in enumerate(set(self.flight_schedule_df["ACFT_Type"]))}

    def extract_aircraft_name(self):
        return {aircraft_name:id for id, aircraft_name in enumerate(set(self.flight_schedule_df["ACFT_REGN"]))}

    def extract_max_min_seating_capacity(self):
        seating_capacity_set = set(self.flight_schedule_df["Seating_Capacity"])

        return {"max":max(seating_capacity_set), "min":min(seating_capacity_set)}

    def extract_max_min_baggage_weight(self):
        baggage_weight = set(self.flight_schedule_df["BaggageWeight_KG"])

        return {"max":max(baggage_weight), "min":min(baggage_weight)}

if __name__ == "__main__":
    import pandas as pd
    import os

    delay_reason_df = pd.read_csv('/Users/jaychan/GitHub/flight-delay-forecasting-model/data/DelayReason.csv')
    flight_schedule_df = pd.read_csv('/Users/jaychan/GitHub/flight-delay-forecasting-model/data/FlightSchedule.csv')
    # print(delay_reason_df)
    # print(flight_schedule_df[12])
    info_prefix = InfoPrefix(flight_schedule_df)
    flight_dataset = FlightDataset(
        flight_schedule_df,
        info_prefix.extract_route(),
        info_prefix.extract_aircraft_type(),
        info_prefix.extract_aircraft_name(),
        info_prefix.extract_max_min_seating_capacity(),
        info_prefix.extract_max_min_baggage_weight()
    )
    print(flight_dataset.__getitem__(12))
    print(flight_dataset.__len__())
    print(flight_dataset.input_dim())

    # print(info_prefix.extract_aircraft_name())
    # print(info_prefix.extract_aircraft_type())
    # print(info_prefix.extract_route())
    # print(info_prefix.extract_max_min_seating_capacity())
    # print(info_prefix.extract_max_min_baggage_weight())