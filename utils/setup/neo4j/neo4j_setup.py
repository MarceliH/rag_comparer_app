import os
import neo4j


def check_neo4j_aura_connection(uri, username, password):
    with neo4j.GraphDatabase.driver(uri, auth=(username, password)) as driver:
        try:
            driver.verify_connectivity()
            return True
        except Exception as e:
            return False
