import os
import argparse
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
        # 尝试验证连接以尽早失败
        with self.driver.session() as session:
            session.run("RETURN 1")

    def close(self):
        # 关闭数据库连接
        self.driver.close()

    def clear_database(self):
        """清除数据库中的所有节点和关系，方便重新导入"""
        with self.driver.session() as session:
            print("正在清除数据库...")
            session.run("MATCH (n) DETACH DELETE n")
            print("数据库已清空。")

    def import_data(self, excel_path=None):
        """从Excel文件导入数据到Neo4j"""
        # 解析Excel路径（支持参数、环境变量与常见默认位置）
        excel_path = resolve_excel_path(excel_path)
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

def resolve_excel_path(excel_path_arg=None):
    """按照优先级解析 Excel 路径，并在常见位置进行回退查找。"""
    # 1) 命令行参数优先
    if excel_path_arg and os.path.isfile(excel_path_arg):
        return excel_path_arg

    # 2) 环境变量
    env_path = os.getenv('TELECOM_EXCEL_PATH')
    if env_path and os.path.isfile(env_path):
        return env_path

    # 3) 常见默认路径集合
    candidates = []
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    cwd = os.getcwd()

    for base in [
        script_dir,
        project_root,
        cwd,
        os.path.join(project_root, 'data'),
        os.path.join(cwd, 'data'),
    ]:
        candidates.append(os.path.join(base, 'telecom_data.xlsx'))

    for path in candidates:
        if os.path.isfile(path):
            print(f"已解析 Excel 路径: {path}")
            return path

    # 未找到，返回一个合理的默认值（供错误消息显示）
    # 默认同脚本目录
    default_path = os.path.join(script_dir, 'telecom_data.xlsx')
    print("未能在常见位置找到 'telecom_data.xlsx'。"
          " 可通过 --excel 或 TELECOM_EXCEL_PATH 指定，"
          f" 或将文件放置到以下任意位置之一: {candidates}")
    return default_path


if __name__ == '__main__':
    # 校验环境变量
    if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
        raise ValueError("Neo4j 数据库连接信息未在 .env 文件中完全设置: 请设置 NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD")

    # 参数与环境变量
    parser = argparse.ArgumentParser(description='Import telecom Excel data into Neo4j.')
    parser.add_argument('--excel', type=str, default=None, help='Path to telecom_data.xlsx')
    parser.add_argument('--clear-first', action='store_true', help='Clear database before import')
    args = parser.parse_args()

    # 允许通过环境变量控制是否清库
    clear_first_env = os.getenv('NEO4J_CLEAR_FIRST', 'false').lower() in ['1', 'true', 'yes']
    clear_first = args.clear_first or clear_first_env

    importer = Neo4jImporter(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

    try:
        if clear_first:
            importer.clear_database()

        importer.import_data(excel_path=args.excel)
    finally:
        importer.close()

