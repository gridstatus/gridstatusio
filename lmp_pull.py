from gridstatusio.gs_client import GridStatusClient
import logging


# Logger setup
def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create console handler and set level to INFO
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add formatter to ch
    ch.setFormatter(formatter)
    
    # Add ch to logger
    logger.addHandler(ch)
    
    return logger

logger = setup_logger()

def pull_lmp(client: GridStatusClient, ba: str, location: str, start_date: str, end_date: str, market_type: str):
    """
    Pull day ahead and real-time LMP data for a list of locations.
    """
    dataset_map = {
        'miso': {'day_ahead': 'miso_lmp_day_ahead_hourly', 'real_time': 'miso_lmp_real_time_5_min_weekly', 'tz': 'US/Central'},  # done
        'spp': {'day_ahead': 'spp_lmp_day_ahead_hourly', 'real_time': 'spp_lmp_real_time_5_min', 'tz': 'US/Central'},
        'ercot': {'day_ahead': 'ercot_spp_day_ahead_hourly', 'real_time': 'ercot_spp_real_time_15_min', 'tz': 'US/Central'}
    }
    
    data = client.get_dataset(
        dataset=dataset_map[ba][market_type],
        start=start_date,
        end=end_date,
            filter_column="location",
            filter_value=location,
            tz=dataset_map[ba]['tz'],
            limit=QUERY_LIMIT,
    )
    return data

if __name__ == "__main__":
    client = GridStatusClient()
    QUERY_LIMIT = 10000

    miso_locations = [
        'ALTW.PIOPRAIR1',
        'AMIL.GDTOWER4',
        'DECO.MCKINLEY1',
        'DECO.MINDEN1',
        'EES.SAN_JC2_CT',
        'MEC.FARMER'
    ]
    
    spp_locations = [
        'BLUECANYON5',
        'EDE_MWW',
        'SPS.GPW1.GREATPLNS',
        'SPS.TNSK.LLANOWIND1UNIT',
        'MPS.ROCKCREEK',
        'NPPD_COOPR'
    ]
    
    ercot_locations = [
        'COTPLNS_RN'
    ]

    locations: dict[str, list[str]] = {
        'miso': miso_locations,
        'spp': spp_locations,
        'ercot': ercot_locations
    }
    
    start_date = "2021-01-01"
    end_date = "2024-10-01"
    
    data = pull_lmp(client, 'miso', 'DECO.MINDEN1', start_date, end_date, 'day_ahead')
    data.write_csv("data/miso_DECO.MINDEN1_day_ahead_2021-01-01_2024-10-01.csv")
    
    # for ba in locations:
    #     for location in locations[ba]:
    #         for market_type in ['day_ahead', 'real_time']:
    #             logger.info(f"Pulling {ba} {location} {market_type} data from {start_date} to {end_date}...")
    #             failed_pulls = []
    #             try:
    #                 data = pull_lmp(client, location, start_date, end_date, ba, market_type)
    #                 data.to_csv(f"data/{ba}_{location}_{market_type}_{start_date}_{end_date}.csv")
    #                 logger.info(f"Pulled {ba} {location} {market_type} data from {start_date} to {end_date} and saved to csv")
    #             except Exception as e:
    #                 logger.error(f"Error pulling {ba} {location} {market_type} data: {e}")
    #                 failed_pulls.append((ba, location, market_type, str(e)))
    #                 continue


            
            

