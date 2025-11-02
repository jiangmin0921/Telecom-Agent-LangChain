import os
import pandas as pd
from neo4j import GraphDatabase
from dotenv import load_dotenv
from tqdm import tqdm  # 用于显示漂亮的进度条

# 加载 .env 文件中的环境变量
load_dotenv()

# 从环境变量获取数据库连接信息
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


class Neo4jImporter:
    def __init__(self, uri, user, password):
        # 初始化数据库驱动
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        # 关闭数据库连接
        self.driver.close()

    def clear_database(self):
        """清除数据库中的所有节点和关系，方便重新导入"""
        with self.driver.session() as session:
            print("正在清除数据库...")
            session.run("MATCH (n) DETACH DELETE n")
            print("数据库已清空。")

    def import_data(self, excel_path='telecom_data.xlsx'):
        """从Excel文件导入数据到Neo4j"""
        try:
            df = pd.read_excel(excel_path)
        except FileNotFoundError:
            print(f"错误: 未找到 '{excel_path}' 文件。请先运行 'generate_data.py'。")
            return

        # 使用 MERGE 语句确保幂等性（重复运行不会创建重复节点）
        # 如果节点已存在，则不会创建；如果不存在，则创建。
        customer_query = """
        MERGE (c:Customer {id: $row.customer_id})
        ON CREATE SET c.name = $row.customer_name, c.phone = $row.phone_number, c.address = $row.address
        """
        plan_query = """
        MERGE (p:Plan {id: $row.plan_id})
        ON CREATE SET p.name = $row.plan_name, p.price = $row.price, 
                      p.data_limit = $row.data_limit, p.voice_limit = $row.voice_limit,
                      p.broadband_speed = $row.broadband_speed
        """
        usage_query = """
        MERGE (u:Usage {customer_id: $row.customer_id, month: $row.usage_month})
        ON CREATE SET u.data_used = $row.data_used, u.voice_used = $row.voice_used
        """
        relationship_query = """
        MATCH (c:Customer {id: $row.customer_id})
        MATCH (p:Plan {id: $row.plan_id})
        MERGE (c)-[:SUBSCRIBED_TO]->(p)
        """

        with self.driver.session() as session:
            print(f"开始从 '{excel_path}' 导入数据到 Neo4j...")
            # 使用tqdm显示进度条
            for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Importing Records"):
                row_dict = dict(row)
                session.run(customer_query, row=row_dict)
                session.run(plan_query, row=row_dict)
                session.run(usage_query, row=row_dict)
                session.run(relationship_query, row=row_dict)

        print("数据导入成功！")

        if __name__ == '__main__':
            importer = Neo4jImporter(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

        # 步骤1: 清空数据库 (可选，如果你想从一个干净的状态开始)
        # 在第一次运行时，建议执行此操作。
        importer.clear_database()

        # 步骤2: 导入数据
        importer.import_data()

        # 步骤3: 关闭连接
        importer.close()

