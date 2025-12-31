# Firmographic Prompt

**任务目标** 基于给定的多源用户数据，推断机构/非个人用户的静态画像，包括机构类型、所属行业和核心职能。

**分析原则** 1. 仔细分析每个数据类别，切勿遗漏，并严格遵循以下原则。  
2. **数据可靠性分析**：始终交叉检查并验证可信度，而不是直接采信。  
3. **严格拒识原则**：不要输出低置信度的推断。如果证据不足，输出“NA”。如果存在多个可能结果，仅返回置信度最高的一个。如果无法确定，返回“NA”。  
4. **特殊数据使用**：  
   - "认证类型" 是平台元数据，具有最高优先级，直接决定“机构类型”。

**详细分析要求** 1. **确定机构类型** 1.1 直接根据认证类型代码 (Verified Type) 映射，无需推理，作为事实依据。  
    1.2 映射规则：  
        - **Government (政府)**: 对应 Verified Type 1  
        - **Enterprise (企业)**: 对应 Verified Type 2  
        - **Media (媒体)**: 对应 Verified Type 3  
        - **Institution (机构)**: 对应 Verified Type 7  

2. **推断行业** 2.1 结合用户的用户名、简介和帖子内容来推断行业。  
    2.2 **行业分类 (Industry)**：  
        "Agriculture and Fishery" (农林牧渔), "Manufacturing" (制造业), "Real Estate and Construction" (建筑与房地产), "Commerce and Retail" (商业与零售), "Transport and Logistics" (交通运输与物流), "High-Tech" (高科技), "Services" (服务业), "Finance" (金融), "Education and Training" (教育培训), "Healthcare" (医疗健康), "Media, Culture, Sports and Entertainment" (传媒文化体娱), "Government and Public Institutions" (政府与公共机构)。

3. **推断账号职能** 3.1 *分析账号在社交媒体上的主要行为意图，从以下 6 类中选择：* - **"News & Policy Info"**: 新闻与政策发布。  
        - **"Brand Image Building"**: 品牌形象建设。  
        - **"Product Promotion & Sales"**: 产品推广与销售。  
        - **"Customer Service"**: 客户服务与互动。  
        - **"Traffic & Entertainment"**: 流量与娱乐。  
        - **"Knowledge & Education"**: 知识科普与教育。  

4. **支撑证据与置信度评分** 4.1 对于每个属性，提供：逐步推理过程、使用的证据引用以及置信度等级。  
    4.2 置信度定义：  
        High: 清晰、强有力的证据；  
        Medium-High: 有一定证据，加上逻辑推演；  
        Medium: 无直接证据但有合理的间接线索；  
        Medium-Low: 线索薄弱，存在较大不确定性；  
        Low: 证据不足，始终直接输出“NA”。  

**输出格式** 请严格按照以下 JSON 格式返回（注意：只返回 JSON，不要包含 Markdown 标记）：  
{{  
  "org_type": {{ "tag": "...", "confidence": "...", "evidence": "..." }},  
  "industry": {{ "tag": "...", "confidence": "...", "evidence": "..." }},  
  "function": {{ "tag": "...", "confidence": "...", "evidence": "..." }}  
}}  

**多源用户信息** "用户昵称": "{username}"  
"个人简介": "{bio}"  
"认证类型": "{verified_type} ({mapped_type_name})"  
"认证详情": "{verified_info}"  
"帖子内容 (Posts)": {posts_content}