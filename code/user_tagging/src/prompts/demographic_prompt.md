# This file contains the demographic prompt template for tagging users based on demographic information.

**任务目标**
基于给定的多源用户数据，推断用户的静态画像，包括年龄、性别、行业、职业、人生阶段和教育水平。

**分析原则**
1. 仔细分析每个数据类别，切勿遗漏，并严格遵循以下原则。
2. **数据可靠性分析**：始终交叉检查并验证可信度，而不是直接采信。
3. **避免先验假设**：例如，不要仅根据年龄就假设婚姻状况、子女情况或就业情况。
4. **特殊数据使用**：
   - “自填性别”是来自平台的高可靠元数据。优先考虑自填性别。
   - “注册时间”说明了账号的存在时长。
   - “帖子内容"中需要区分 `content` (核心表达) 和 `quote_content` (背景信息)。

**详细分析要求**

1. **推断年龄和性别**
    1.1 结合用户信息推断年龄和性别，参考**自填性别**来确定最终性别。
    1.2 年龄区间：0-18, 18-23, 24-30, 31-40, 41-50, 50+。
    1.3 性别分类：Male, Female。

2. **推断行业和职业**
    2.1 结合用户的用户名、简介和帖子内容来推断行业和职业。
    2.2 **行业分类 (Industry)**：
        "Agriculture and Fishery" (农林牧渔), "Manufacturing" (制造业), "Real Estate and Construction" (建筑与房地产), "Commerce and Retail" (商业与零售), "Transport and Logistics" (交通运输与物流), "High-Tech" (高科技), "Services" (服务业), "Finance" (金融), "Education and Training" (教育培训), "Healthcare" (医疗健康), "Media, Culture, Sports and Entertainment" (传媒文化体娱), "Government and Public Institutions" (政府与公共机构), "Not Employed" (未就业)。
    2.3 **职业分类 (Occupation)**：
        "Software" (软件/互联网), "Clerical Staff" (文员/行政), "Education and Trainer" (教育/培训师), "Beauty and Hairdressing" (美容美发), "Skilled Workers" (技术工人), "Government and Public Sector" (公务员/事业单位), "Transportation and Logistics" (交通物流人员), "Hospitality and Entertainment" (酒店/娱乐服务), "Media and Culture" (媒体文化工作者), "Independent Media" (自媒体), "Healthcare" (医护人员), "Agriculture and Fishery" (农林牧渔人员), "Finance and Insurance" (金融保险从业者), "Self-Employed" (个体经营者), "Domestic and Security" (家政安保), "Student" (学生), "High-Tech Hardware" (硬件工程师), "Retiree" (退休人员), "Homemaker" (家庭主妇/夫)。

3. **推断人生阶段**
    3.1 结合用户信息进行推断，参考推断年龄和注册时间。
    3.2 请从以下选项中选择：
        "Single" (单身), "In Relationship" (恋爱中), "Pre-Marital" (备婚/订婚), "Married, No Children" (已婚未育), "Pre-Pregnancy and Pregnancy" (备孕/怀孕), "Parenting (Child 0–2)" (育儿期 0-2岁), "Parenting (Child 3-5)" (育儿期 3-5岁), "Parenting (Child 6-11)" (育儿期 6-11岁), "Parenting (Child 12-14)" (育儿期 12-14岁), "Parenting (Child 15-17)" (育儿期 15-17岁), "Parenting (Adult Child)" (育儿期-成年子女), "Parenting (Child Age Unknown)" (育儿期-子女年龄未知)。

4. **推断教育水平**
    4.1 请从以下选项中选择：
        "Junior High or Below" (初中及以下), "Senior High or Vocational" (高中/中专/职高), "Bachelor's or Associate" (本科/大专), "Postgraduate or Above" (研究生及以上)。
5. **支撑证据与置信度评分**
    5.1 对于每个属性，提供：逐步推理过程、使用的证据引用以及置信度等级。
    5.2 置信度定义：
        High: 清晰、强有力的证据；
        Medium-High: 有一定证据，加上逻辑推演；
        Medium: 无直接证据但有合理的间接线索；
        Medium-Low: 线索薄弱，存在较大不确定性；
        Low: 证据不足，直接输出“NA”，

**输出格式**
请严格按照以下JSON格式返回：
{{
  "age": {{ "tag": "...", "confidence": "...", "evidence": "..." }},
  "gender": {{ "tag": "...", "confidence": "...", "evidence": "..." }},
  "industry": {{ "tag": "...", "confidence": "...", "evidence": "..." }},
  "occupation": {{ "tag": "...", "confidence": "...", "evidence": "..." }},
  "life_stage": {{ "tag": "...", "confidence": "...", "evidence": "..." }},
  "education_level": {{ "tag": "...", "confidence": "...", "evidence": "..." }}
}}

**多源用户信息**
"用户昵称 (User Name)": "{username}"
"帖子内容 (Posts)": {posts_content}
"个人简介 (Bio)": {bio}
"自填性别 (Self-reported Gender)": "{gender_reported}"
"注册时间 (Registration Time)": "{reg_time}"
"地理位置 (Region)": "{location_reported}"
"认证信息 (Verified Info)": "{verified_info}"