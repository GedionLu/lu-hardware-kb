#!/usr/bin/env python3
"""Rewrite query.py with improved INTENT_MAP"""
import re

with open('query.py') as f:
    content = f.read()

old_start = content.find("INTENT_MAP = {")
old_end = content.find("\n\n", content.find("'general_setup'")) + 2

new_intent_map = """INTENT_MAP = {
    'usb_connect': {
        'keywords': ['USB', '电脑', '连接', '线缆', '键盘口'],
        'subcategories': ['restore_factory', 'interface', 'test_code', 'setup'],
        'desc': 'USB连接、线缆、电脑通信',
        'examples': ['1900 怎么连电脑', 'HH760 USB连接'],
    },
    'serial_connect': {
        'keywords': ['串口', 'RS232', 'serial', 'COM', 'com口'],
        'subcategories': ['interface', 'setup', 'restore_factory'],
        'desc': '串口连接、RS232、COM口通信',
        'examples': ['1902 串口怎么连', '7680g RS232设置'],
    },
    'bluetooth_pairing': {
        'keywords': ['配对', '蓝牙', '无线', '底座'],
        'subcategories': ['pairing', 'setup'],
        'desc': '蓝牙配对、无线连接、底座配对',
        'examples': ['1902 蓝牙配对', 'HH762 无线连接'],
    },
    'virtual_com_port': {
        'keywords': ['虚拟串口', 'USB虚拟串口', 'USB仿真串口', '转串口'],
        'subcategories': ['interface', 'setup', 'restore_factory'],
        'desc': 'USB虚拟串口、驱动安装',
        'examples': ['1900 虚拟串口', 'OH430 USB转串口'],
    },
    'add_suffix': {
        'keywords': ['回车', '换行', '后缀', 'CRLF', '自动换行', '自动回车'],
        'subcategories': ['suffix', 'setup'],
        'desc': '加回车换行、后缀设置',
        'examples': ['OH430 加回车', '1900 自动换行'],
    },
    'add_prefix': {
        'keywords': ['前缀', '自定义前缀', '加前缀'],
        'subcategories': ['prefix', 'setup'],
        'desc': '加前缀、自定义前缀',
        'examples': ['OH430 加前缀', '1902 设置前缀'],
    },
    'restore_factory': {
        'keywords': ['恢复出厂', '重置', '初始化', '恢复默认', '清空'],
        'subcategories': ['restore_factory'],
        'desc': '恢复出厂设置、重置、初始化',
        'examples': ['HH490 恢复出厂', '1902 初始化'],
    },
    'test_comm': {
        'keywords': ['测试', '通信验证', '没反应', '扫不出', '不好使', '不行'],
        'subcategories': ['test_code', 'restore_factory'],
        'desc': '测试通信、故障排查、扫不出来',
        'examples': ['扫描枪没反应', 'HH760 扫不出来'],
    },
    'chinese_qr': {
        'keywords': ['中文', '中文字符', '汉字'],
        'subcategories': ['interface', 'setup', 'feature'],
        'desc': '读取含中文的二维码',
        'examples': ['1900 扫中文二维码', '1472 中文输出'],
    },
    'data_format': {
        'keywords': ['截取', '格式编辑', '数据替换', '序列扫描', '只输出', '过滤'],
        'subcategories': ['feature', 'setup'],
        'desc': '数据格式编辑、截取字段、替换',
        'examples': ['1902 截取后8位', '1952 格式编辑'],
    },
    'continuous_scan': {
        'keywords': ['连续模式', '自动扫描', '常亮', '持续'],
        'subcategories': ['feature', 'setup'],
        'desc': '连续扫描模式、自动扫描',
        'examples': ['OH430 连续模式', '7680g 常亮'],
    },
    'dpm_optimize': {
        'keywords': ['DPM', '点阵', '刻字码', '金属'],
        'subcategories': ['feature', 'setup'],
        'desc': 'DPM码读取优化、点阵码、昏暗环境',
        'examples': ['190x DPM优化', '昏暗环境读码'],
    },
    'firmware': {
        'keywords': ['固件', '升级', '版本'],
        'subcategories': ['feature'],
        'desc': '固件升级、版本查看',
        'examples': ['PM43 固件版本', 'Fiji 固件升级'],
    },
    'no_read': {
        'keywords': ['No Read', '无读取', '空扫'],
        'subcategories': ['feature', 'setup'],
        'desc': 'No Read模式、空扫设置',
        'examples': ['HH490 No Read', '关闭No Read'],
    },
    'interface_mode': {
        'keywords': ['接口码', '接口模式', '接口设置'],
        'subcategories': ['interface', 'setup', 'restore_factory'],
        'desc': '接口码、接口模式设置',
        'examples': ['1900 接口码', '19xx 接口设置'],
    },
    'pairing': {
        'keywords': ['配对', '配对码', '蓝牙连接'],
        'subcategories': ['pairing', 'setup'],
        'desc': '配对码、重新配对',
        'examples': ['1902 配对码', 'HH492 重新配对'],
    },
    'general_setup': {
        'keywords': ['设置', '配置', '使用', '怎么用', '说明', '操作', '调试'],
        'subcategories': ['restore_factory', 'interface', 'test_code', 'setup'],
        'desc': '以上都不匹配时的兜底选项',
        'examples': [],
    },
}

FEW_SHOT_EXAMPLES = []
for _iname, _icfg in sorted(INTENT_MAP.items()):
    for ex in _icfg.get('examples', []):
        FEW_SHOT_EXAMPLES.append((ex, _iname))
"""

content = content[:old_start] + new_intent_map + content[old_end:]

with open('query.py', 'w') as f:
    f.write(content)

print("INTENT_MAP updated with desc + examples")
