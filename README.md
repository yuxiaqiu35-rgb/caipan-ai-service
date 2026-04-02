# 财判AI · 算法服务

财务舞弊风险识别系统的算法后端，为Coze工作流提供HTTP计算接口。

## 接口列表

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 健康检查 |
| `/api/tukey_detection` | POST | Tukey箱线图异常检测 |
| `/api/isolation_forest` | POST | Isolation Forest多维异常检测 |
| `/api/rule_engine` | POST | 预设规则引擎 |
| `/api/text_analysis` | POST | 中文三维可读性检测 |
| `/api/causal_analysis` | POST | 因果方向分析（初赛版） |
| `/api/f_score` | POST | Dechow F-Score计算 |
| `/api/ds_aggregation` | POST | D-S证据理论聚合 |
| `/api/fetch_financial_data` | POST | AKShare财务数据获取 |

## 本地运行

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

访问 http://localhost:8000/docs 查看交互式API文档

## Railway部署

推送到GitHub后，Railway自动检测并部署。
