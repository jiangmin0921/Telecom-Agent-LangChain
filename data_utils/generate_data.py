import pandas as pd
from faker import Faker
import random

# 初始化Faker，用于生成模拟数据
fake = Faker('zh_CN')  # 使用中文数据

# 预定义一些电信套餐，让数据更真实
PLANS = [
    {'plan_id': 'P01', 'plan_name': '5G畅享套餐', 'price': 129, 'data_limit': 40, 'voice_limit': 500,
     'broadband_speed': None},
    {'plan_id': 'P02', 'plan_name': '4G流量王套餐', 'price': 79, 'data_limit': 20, 'voice_limit': 200,
     'broadband_speed': None},
    {'plan_id': 'P03', 'plan_name': '家庭宽带融合套餐', 'price': 199, 'data_limit': 60, 'voice_limit': 1000,
     'broadband_speed': 300},
    {'plan_id': 'P04', 'plan_name': '校园青春套餐', 'price': 59, 'data_limit': 30, 'voice_limit': 100,
     'broadband_speed': None},
    {'plan_id': 'P05', 'plan_name': '商务尊享套餐', 'price': 299, 'data_limit': 100, 'voice_limit': 2000,
     'broadband_speed': 500},
]

# 要生成的数据条数
NUM_RECORDS = 500


def generate_telecom_data(num_records):
    """生成指定数量的电信模拟数据"""
    data = []
    for i in range(num_records):
        customer_id = f'C{i + 1:04d}'  # 生成 C0001, C0002... 格式的ID
        customer_name = fake.name()
        phone_number = fake.phone_number()
        address = fake.address()

        # 为每个客户随机分配一个套餐
        assigned_plan = random.choice(PLANS)

        # 根据套餐限额生成合理的使用数据
        data_used = round(random.uniform(assigned_plan['data_limit'] * 0.5, assigned_plan['data_limit'] * 1.2), 2)
        voice_used = round(random.uniform(assigned_plan['voice_limit'] * 0.5, assigned_plan['voice_limit'] * 1.2), 0)

        record = {
            'customer_id': customer_id,
            'customer_name': customer_name,
            'phone_number': phone_number,
            'address': address,
            'plan_id': assigned_plan['plan_id'],
            'plan_name': assigned_plan['plan_name'],
            'price': assigned_plan['price'],
            'data_limit': assigned_plan['data_limit'],
            'voice_limit': assigned_plan['voice_limit'],
            'broadband_speed': assigned_plan['broadband_speed'],
            'usage_month': '2025-10',  # 假设都是10月份的数据
            'data_used': data_used,
            'voice_used': int(voice_used),
        }
        data.append(record)

    return pd.DataFrame(data)


if __name__ == "__main__":
    print(f"正在生成 {NUM_RECORDS} 条模拟数据...")
    df = generate_telecom_data(NUM_RECORDS)

    # 将DataFrame保存到Excel文件
    output_filename = 'telecom_data.xlsx'
    df.to_excel(output_filename, index=False, engine='openpyxl')

    print(f"数据生成完毕！已保存到 '{output_filename}' 文件中。")
    print("文件预览(前5行): ")
    print(df.head())

