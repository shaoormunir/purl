from pathlib import Path

from custom_command import LinkCountingCommand
from openwpm.command_sequence import CommandSequence
from openwpm.commands.browser_commands import GetCommand
from openwpm.config import BrowserParams, ManagerParams
from openwpm.storage.sql_provider import SQLiteStorageProvider
from openwpm.storage.leveldb import LevelDbProvider
from openwpm.task_manager import TaskManager

from tqdm.auto import tqdm

import pandas as pd

# The list of sites that we wish to crawl

NUM_BROWSERS = 4

starting_index = 0

LOG_FILE = "multi-crawl.log"


df = pd.read_csv('sites.csv')

# open the log file in append mode
with open(LOG_FILE, "a") as log_file:
    for i in range(starting_index, 10000, 1000):
        log_file.write(f"Starting crawl {i} to {i+1000}\n")
        log_file.flush()

        if i != 9000:
            sites  = df['url'][i:i+1000].tolist()
        else:
            sites = df['url'][i:].tolist()

        # Loads the default ManagerParams
        # and NUM_BROWSERS copies of the default BrowserParams

        manager_params = ManagerParams(num_browsers=NUM_BROWSERS)
        browser_params = [BrowserParams(display_mode="headless") for _ in range(NUM_BROWSERS)]

        # Update browser configuration (use this for per-browser settings)
        for browser_param in browser_params:
            # Record HTTP Requests and Responses
            browser_param.http_instrument = True
            # Record cookie changes
            browser_param.cookie_instrument = True
            # Record Navigations
            browser_param.navigation_instrument = True
            # Record JS Web API calls
            browser_param.js_instrument = True
            # Record the callstack of all WebRequests made
            browser_param.callstack_instrument = True
            # Record DNS resolution
            browser_param.dns_instrument = True
            # save the javascript files
            browser_param.save_content = "script"
            # allow third party cookies
            browser_param.tp_cookies = 'never'
            # Prevent any response by server due to bot detection
            browser_param.bot_mitigation  = True

        # Update TaskManager configuration (use this for crawl-wide settings)
        manager_params.data_directory = Path(f"./datadir-{i}/")
        manager_params.log_path = Path(f"./datadir-{i}/openwpm.log")

        # memory_watchdog and process_watchdog are useful for large scale cloud crawls.
        # Please refer to docs/Configuration.md#platform-configuration-options for more information
        # manager_params.memory_watchdog = True
        # manager_params.process_watchdog = True


        # Commands time out by default after 60 seconds
        with TaskManager(
            manager_params,
            browser_params,
            SQLiteStorageProvider(Path(f"./datadir-{i}/crawl-data.sqlite")),
            LevelDbProvider(Path(f"./datadir-{i}/content.ldb")),
        ) as manager:
            # Visits the sites
            for index, site in enumerate(tqdm(sites)):

                def callback(success: bool, val: str = site) -> None:
                    print(
                        f"CommandSequence for {val} ran {'successfully' if success else 'unsuccessfully'}"
                    )

                # Parallelize sites over all number of browsers set above.
                command_sequence = CommandSequence(
                    site,
                    site_rank=index,
                    callback=callback,
                )

                # Start by visiting the page
                command_sequence.append_command(GetCommand(url=site, sleep=3), timeout=60)
                # Have a look at custom_command.py to see how to implement your own command
                command_sequence.append_command(LinkCountingCommand())

                # Run commands across all browsers (simple parallelization)
                manager.execute_command_sequence(command_sequence)
        
        log_file.write(f"Finished crawl {i} to {i+1000}\n")
        log_file.flush()