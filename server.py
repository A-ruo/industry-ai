import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from typing import Optional

app = FastAPI(title="行业分析AI后端")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    raise ValueError("请先设置环境变量 DEEPSEEK_API_KEY")
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

class IndustryRequest(BaseModel):
    industry_name: str
    user_data: Optional[str] = None
    analysis_mode: Optional[str] = "quick"

class CompanyRequest(BaseModel):
    company_name: str
    user_data: Optional[str] = None
    analysis_mode: Optional[str] = "quick"

class ReportRequest(BaseModel):
    content: str
    company_name: str

# ---------- 提示词 ----------
ANALYSIS_PROMPT_TEMPLATE = """
你是一位资深证券行业分析师，擅长用《如何快速了解一个行业》中的“四步一核心”框架进行结构化分析。

**核心原则（必须严格遵守）**：
1. 先定量后定性：先列出具体数据，再用文字解释。所有重要判断必须基于数据趋势，而非泛泛而谈。
   你的训练数据可能有截止日期，如果遇到不确定或过时的具体数字，必须明确标注“数据暂缺，建议查询XX来源”，不可编造。
2. 先外部后内部：先分析政策、技术、经济等外部驱动因素，再谈企业优势与竞争格局。
3. 为景气度相关变量设定具体的数值阈值（高-中-低），不可只写“较高”“一般”等模糊描述。如果某项数据确实无法获取，明确标注“数据暂缺”，并给出替代观察指标。
4. 输出使用Markdown格式，但**只能使用二级标题（## 标题）和列表（- 或 1.）**。禁止使用三级及以下标题（###、####等）。所有需要分层的内容，用**粗体**作为小标题，后面跟列表项。
5. 如果需要呈现对比数据，可以使用Markdown表格，但表格必须严格对齐，每列用`|`分隔，表头与内容用`|---|---|`分隔。
6. 报告末尾必须附加一个JSON代码块（用```json包裹），包含指定的结构化指标，用于系统提取展示。

请针对【{industry_name}】行业，严格按以下结构输出深度分析报告：

## 一、行业框架（八大维度速览）
必须使用 Markdown 表格呈现，格式如下：
| 维度 | 概念 | 定义 | 关键指标 |
| --- | --- | --- | --- |
| 行业生命周期 | ... | ... | ... |
| 商业模式 | ... | ... | ... |
| 市场规模 | ... | ... | ... |
| 竞争格局 | ... | ... | ... |
| 驱动因素 | ... | ... | ... |
| 景气度 | ... | ... | ... |
| 壁垒 | ... | ... | ... |
| 盈利模式 | ... | ... | ... |
表格后无需再重复文字说明。

## 二、关键数据（4×3 信息卡片）
**行业层面**
- **渗透率**：（当前值及近3年趋势，若暂缺请注明）
- **CR4 集中度**：（前四名企业市占率合计，近3年变化）
- **年复合增速**：（过去3-5年CAGR）

**公司层面（选取3-5家头部企业）**
每家列出：
- **营收结构**：（按产品/业务线拆分）
- **毛利率**：（近三年变化方向及幅度）
- **费用率**：（销售/管理/研发费率）

**外部层面**
每类列出3个关键变量，标注影响方向（↑有利 ↓不利 →中性/待观察）：
- **政策**×3：
- **技术**×3：
- **经济周期**×3：

**痛点层面**
- **客户投诉高频词**×3：
- **产能利用率**：（当前值及趋势）
- **库存周期**：（当前所处阶段）

## 三、景气度判断
必须使用 Markdown 表格呈现，格式如下：
| 指标类别 | 关键变量 | 当前实际值 | 高-中-低阈值 | 当前档位 | 趋势方向 | 变化幅度 |
| --- | --- | --- | --- | --- | --- | --- |
| 需求端 | ... | ... | ... | ... | ... | ... |
| 供给端 | ... | ... | ... | ... | ... | ... |
| 价格端 | ... | ... | ... | ... | ... | ... |
| 盈利端 | ... | ... | ... | ... | ... | ... |
表格后另起一段给出**景气度总体档位**（高/中/低）及未来1-3个月预判。

## 四、洞见与建议
**行业定位**
一句话说明处于哪一生命周期，核心解决什么痛点。
**关键变化**
列出最近值得关注的关键趋势，标注方向（Direction）和幅度（Magnitude）。
**投资/决策建议**
给出“布局-观望-撤退”三选一，并附两条以上具体理由。

```json
{{
  "industry_name": "{industry_name}",
  "lifecycle_stage": "导入期/成长期/成熟期/衰退期",
  "penetration_rate": "当前渗透率百分比",
  "market_size": "市场规模",
  "yoy_growth": "同比增速",
  "cr4": "前四名集中度百分比",
  "gross_margin_range": "毛利率区间",
  "prosperity_level": "高/中/低",
  "prosperity_trend": "↑/↓/→",
  "investment_advice": "布局/观望/撤退"
}}

请直接输出报告正文和JSON代码块，不要任何开场白、问候语或确认语句（如“好的”、“作为...”、“根据...”），也不要使用三级标题（###）。
"""

COMPANY_ANALYSIS_PROMPT_TEMPLATE = """
你是一位资深证券分析师，擅长对上市公司进行全方位深度剖析。请严格按照以下15个维度对【{company_name}】进行分析，使用Markdown格式输出，每个维度用"## 维度名称"作为大标题，内部用列表或段落展开。

**核心原则**：
1. 先定量后定性：每个维度尽量给出具体数字（如毛利率36.05%、增速+31.78%），再分析含义。
   你的训练数据可能有截止日期，如果遇到不确定或过时的具体数字，必须明确标注“数据暂缺，建议查询XX来源”，不可编造。
2. 使用"因为…所以…"的因果逻辑链条，而非简单罗列事实。
3. 如果某个维度的数据确实无法获取，请明确标注"数据暂缺"，并给出替代观察建议。
4. 只能使用二级标题（##），禁止使用三级及以下标题（###）。需要分层时，用**粗体小标题**+列表。
5. 在“行业框架”和“景气度判断”板块必须使用Markdown表格呈现数据，表格必须严格对齐，每列用`|`分隔，表头与内容用`|---|---|`分隔。须附加一个JSON代码块，包含关键指标摘要。

## 一、企业画像
用**粗体小标题**分项说明，最后用一段话概括：
- **公司全称与定位**：一句话说明它是谁、做什么的
- **核心业务与产品线**：主要产品系列及营收占比（如有）
- **经营模式**：事业部制/职能制？经销为主还是直营为主？
- **核心业绩驱动因素**：公司自己认为的增长引擎是什么
- **行业环境概述**：所处行业的宏观环境、供需状况、政策影响
- **一句话概括**：总结这家公司的核心特征和行业地位

## 二、核心指标解读
以"虽然…但是…因此…"的逻辑串联以下指标：
- **营业收入**：本期值、同比增速、一句话含义
- **毛利率**：本期值、同比变化、变化原因（产品结构/成本控制/提价）
- **研发费用**：本期值、占营收比、同比增速、投入方向
- **扣非净利润**：本期值、同比增速、与净利润的差异原因
- **结论与展望**：将以上四点串成一个完整故事，给出前瞻性判断

## 三、竞争格局
选取2-3家主要竞争对手，从以下维度对比：
- **市场地位**：行业排名、份额对比
- **核心产品与品牌**：各自的王牌产品
- **渠道策略**：线下/线上布局差异
- **优势与劣势**：各自的核心长板和短板
- **形成你的判断**：用一段话总结行业竞争格局，指出标的公司的护城河所在

## 四、新闻解读
选取一条近期重要新闻（产品发布/战略合作/投资并购等），分析：
- **新闻内容摘要**：发生了什么
- **战略方向判断**：这条新闻反映了公司怎样的战略意图
- **关键举措**：公司具体做了什么
- **影响分析**：短期和长期分别对公司是利好还是利空，为什么
- **结论**：这件事的成功概率和跟踪要点

## 五、估值分析
- **当前估值水平**：动态PE、股价、所在行业分位
- **与竞争对手对比**：列出2-3家对手的PE，判断公司估值是折价还是溢价
- **原因探究**：为什么市场给这个估值？（规模与增速权衡/业务成熟度/市场情绪等）
- **估值修复的催化剂**：什么因素可能触发估值提升（第二曲线/盈利质量/分红回购）
- **最终观点**：当前估值是否合理，是否存在低估或高估

## 六、客户分析
必须使用 Markdown 表格呈现，格式如下：
| 客户类型 | 细分群体 | 需求特点 | 公司渠道能力/优势 |
| --- | --- | --- | --- |
| B2B直接客户 | ... | ... | ... |
| B2C最终用户 | ... | ... | ... |
表格后另起一段给出**结论**：客户结构决定了公司需要怎样的核心能力。

## 七、产品独特性
- **技术领先维度**：公司掌握了哪些行业关键技术？有何专利或独家工艺？
- **成本领先维度**：规模效应、采购议价力、生产效率如何体现？
- **二者协同**：技术和成本如何形成"增强回路"？
- **结论**：公司产品竞争力的核心是什么

## 八、销售渠道
- **渠道全景**：线下经销、线下直营、线上直营、O2O等各自占比和增速
- **渠道战略解读**：公司是"精耕"还是"开拓"？有无渠道优化信号？
- **新兴渠道的战略意义**：会员店、电商、直播等新渠道的角色
- **结论**：渠道力是否是公司的核心竞争力

## 九、护城河分析
从以下四个维度展开，并说明它们如何相互作用：
- **规模与成本壁垒**
- **渠道与网络壁垒**
- **品牌与心智壁垒**
- **技术与创新壁垒**
- **协同效应**：这四大壁垒如何形成自我强化的闭环
- **结论**：护城河的宽度、深度和可持续性

## 十、商业模式画布
用文字描述商业模式的九个要素（可简化为关键要素）：
- **价值主张**：公司为客户创造什么价值
- **客户细分**：服务哪些人群/企业
- **渠道通路**：如何触达客户
- **客户关系**：如何维护客户
- **收入来源**：主要怎么赚钱
- **核心资源**：最关键的资产是什么
- **关键业务**：日常在做什么
- **重要合作**：与谁协同
- **成本结构**：最大的成本项是什么

## 十一、盈利能力
选取关键盈利指标（毛利率、扣非净利率、净利率、营业利润率），逐个分析：
- **本期值、同比变化、变化原因**
- **与外部信息关联**：产品结构、费用控制、非经常性损益等
- **综合判断**：核心业务盈利质量如何，短期利润波动是结构性还是周期性

## 十二、成长能力
- **营收增长率**：趋势及驱动因素
- **净利润增长率**：与扣非净利润的差异原因（重点核查非经常性损益）
- **营业利润增长率**：主营业务的真实成长性
- **核心结论**：公司的成长是"真成长"还是"数字游戏"

## 十三、营运能力
- **存货周转率**：变化趋势、反映的销售与管理效率
- **应收账款周转率**：变化趋势、反映的议价能力和回款管理
- **综合解读**：将营运能力与盈利能力关联，判断公司是否实现"高利润+高周转"

## 十四、偿债能力
- **资产负债率**：长期偿债能力变化及原因
- **流动比率**：短期偿债压力及成因
- **综合判断**：偿债指标变化是经营恶化还是主动战略选择（结合业务扩张、研发投入等解释）

## 十五、杜邦分析
- **ROE**：本期的净资产收益率
- **净利润率**：盈利能力贡献
- **总资产周转率**：营运效率贡献
- **权益乘数**：财务杠杆贡献
- **ROE驱动因素判断**：公司是靠"高利润"还是"高杠杆"还是"高周转"驱动ROE
- **商业叙事**：用杜邦三要素串成一个完整故事
- 用**粗体标题+同一行内容**的形式输出：
  **ROE驱动因素判断**：伊利是典型的“高利润+高周转”驱动型，财务杠杆贡献较小……
  **商业叙事**：该公司……
  注意：标题和内容必须在同一行，不要换行。
```json
{{
  "company_name": "{company_name}",
  "industry_position": "行业地位（龙头/一线/区域龙头等）",
  "revenue_growth": "营收同比增速",
  "gross_margin": "毛利率",
  "deducted_net_profit_growth": "扣非净利润增速",
  "roe": "ROE",
  "pe": "当前动态PE",
  "prosperity_level": "综合质地评估（优秀/良好/一般/较差）",
  "investment_advice": "布局/观望/撤退"
}}
请务必在报告末尾包含完整的 JSON 代码块，字段必须齐全，不可省略。
请直接输出报告正文和JSON代码块，不要任何开场白、问候语或确认语句（如“好的”、“作为...”、“根据...”），也不要使用三级标题（###）。
"""

REPORT_PROMPT_TEMPLATE = """
你是一位资深证券分析师。下面是对【{company_name}】的前15个维度的详细分析内容。请基于这些内容，撰写一份完整的、深度综合投资分析报告，字数不少于2000字。报告必须包含以下部分：

## 一、执行摘要
简洁概括公司核心优势、关键挑战、前景展望和最终投资建议。

## 二、公司概述与行业地位
结合企业画像和竞争格局进行阐述。

## 三、核心战略分析
从商业模式、产品独特性、护城河等维度提炼公司的战略逻辑。

## 四、运营与财务表现概要
整合核心指标、盈利能力、成长能力、营运能力和偿债能力的核心发现，形成连贯分析。

## 五、投资亮点与风险提示
列出3-5个最主要的投资亮点和3-5个核心风险。

## 六、估值与投资建议
给出明确的估值判断（是否低估/合理/高估）和最终投资建议（布局/观望/撤退），并附详细理由。
请直接输出报告正文，不要任何开场白、问候语、确认语句（如“好的”、“作为...”）。
以下是已有的分析内容：
{content}
"""

# ---------- 行业分析接口 ----------
@app.post("/analyze")
async def analyze_industry(request: IndustryRequest):
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="API Key 未配置")

    user_context = ""
    if request.user_data:
        user_context = (
            "**重要：以下是用户提供的真实数据，请优先基于这些数据进行分析，"
            "不要使用你训练数据中的旧信息。如果数据不完整，请基于已有数据分析，"
            "并标注哪些结论基于用户数据、哪些为合理推测。**\n\n"
            f"用户数据：\n{request.user_data}\n\n---\n\n"
        )

    if request.analysis_mode == "quick":
        prompt_template = """
你是一位资深行业分析师，请用最精炼的语言快速概述【{industry_name}】行业，严格遵循以下格式：

## 生命周期
（一句话说明所处阶段及特征）

## 市场规模
（当前规模及近一年增速，仅给出关键数字）

## 竞争格局
（CR4集中度及头部公司名，一句话概括格局）

## 景气度
（当前景气档位及未来1-3个月趋势，一句话）

## 投资建议
（布局/观望/撤退，附一句话理由）

```json
{{
  "industry_name": "{industry_name}",
  "lifecycle_stage": "导入期/成长期/成熟期/衰退期",
  "penetration_rate": "数据暂缺",
  "market_size": "数据暂缺",
  "yoy_growth": "数据暂缺",
  "cr4": "数据暂缺",
  "gross_margin_range": "数据暂缺",
  "prosperity_level": "高/中/低",
  "prosperity_trend": "↑/↓/→",
  "investment_advice": "布局/观望/撤退"
}}
"""
        prompt = user_context + prompt_template.format(industry_name=request.industry_name)
    elif request.analysis_mode == "prosperity":
        prompt_template = """
你是一位景气度分析专家。请针对【{industry_name}】行业，只分析景气度相关内容，包括：
1. 需求端、供给端、价格端、盈利端的关键指标
2. 为每个指标设定高-中-低阈值
3. 给出当前景气度总体档位和未来1-3个月预判
请直接输出报告正文和JSON代码块，不要任何开场白、问候语或确认语句（如“好的”、“作为...”、“根据...”），也不要使用三级标题（###）。

```json
{{
  "industry_name": "{industry_name}",
  "lifecycle_stage": "数据暂缺",
  "penetration_rate": "数据暂缺",
  "market_size": "数据暂缺",
  "yoy_growth": "数据暂缺",
  "cr4": "数据暂缺",
  "gross_margin_range": "数据暂缺",
  "prosperity_level": "高/中/低",
  "prosperity_trend": "↑/↓/→",
  "investment_advice": "数据暂缺"
}}
"""
        prompt = user_context + prompt_template.format(industry_name=request.industry_name)
    else:  # deep
        prompt = user_context + ANALYSIS_PROMPT_TEMPLATE.format(industry_name=request.industry_name)

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            stream=False
        )
        if response.choices[0].message.content:
            ai_output = response.choices[0].message.content
            return {"status": "success", "report": ai_output}
        else:
            raise HTTPException(status_code=500, detail="DeepSeek 返回内容为空")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- 企业分析接口 ----------
@app.post("/analyze_company")
async def analyze_company(request: CompanyRequest):
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="API Key 未配置")

    user_context = ""
    if request.user_data:
        user_context = (
            "**重要：以下是用户提供的真实数据，请优先基于这些数据进行分析，"
            "不要使用你训练数据中的旧信息。如果数据不完整，请基于已有数据分析，"
            "并标注哪些结论基于用户数据、哪些为合理推测。**\n\n"
            f"用户数据：\n{request.user_data}\n\n---\n\n"
        )

    if request.analysis_mode == "quick":
        prompt_template = """
你是一位资深分析师，请用最精炼的语言快速概述【{company_name}】公司，包含以下维度，每个维度仅用1-2句话：

## 企业画像
（一句话定位及核心业务）

## 核心指标
（营收、毛利率、扣非净利的关键数字）

## 竞争格局
（主要对手及市场地位）

## 护城河
（最核心的壁垒，一句话）

## 估值与建议
（当前PE及投资建议）

```json
{{
  "company_name": "{company_name}",
  "industry_position": "行业地位",
  "revenue_growth": "数据暂缺",
  "gross_margin": "数据暂缺",
  "deducted_net_profit_growth": "数据暂缺",
  "roe": "数据暂缺",
  "pe": "数据暂缺",
  "prosperity_level": "数据暂缺",
  "investment_advice": "布局/观望/撤退"
}}
"""
        prompt = user_context + prompt_template.format(company_name=request.company_name)
    elif request.analysis_mode == "prosperity":
        raise HTTPException(status_code=400, detail="企业分析暂不支持景气度专项模式")
    else:
        prompt = user_context + COMPANY_ANALYSIS_PROMPT_TEMPLATE.format(company_name=request.company_name)

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            stream=False
        )
        if response.choices[0].message.content:
            ai_output = response.choices[0].message.content
            return {"status": "success", "report": ai_output}
        else:
            raise HTTPException(status_code=500, detail="DeepSeek 返回内容为空")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- 生成综合报告接口 ----------
@app.post("/generate_company_report")
async def generate_company_report(request: ReportRequest):
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="API Key 未配置")

    prompt = REPORT_PROMPT_TEMPLATE.format(
        company_name=request.company_name,
        content=request.content
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            stream=False
        )
        if response.choices[0].message.content:
            ai_output = response.choices[0].message.content
            return {"status": "success", "report": ai_output}
        else:
            raise HTTPException(status_code=500, detail="DeepSeek 返回内容为空")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)