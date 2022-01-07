import core_logic.htr_word.recognition.misc.jpn.utils.hira_kana_label_1 as labels
import core_logic.htr_word.recognition.misc.jpn.utils.hira_kana_label_1_main as labels1

# define possible character list
hiragana=["あ","い","う","え","お","か","が","き","ぎ","く","ぐ","け","げ","こ","ご","さ","ざ","し","じ","す","ず","せ","ぜ","そ","ぞ","た","だ","ち","ぢ","つ","づ","て","で","と","ど","な","に","ぬ","ね","の","は","ば","ぱ","ひ","び","ぴ","ふ","ぶ","ぷ","へ","べ","ぺ","ほ","ぼ","ぽ","ま","み","む","め","も","や","ゆ","よ","ら","り","る","れ","ろ","わ","を","ん"]

katakana=["ア","ヰ","イ","ウ","ヱ","エ","オ","カ","キ","ク","ケ","コ","サ","シ","ス","セ","ソ","タ","チ","ツ","テ","ト","ナ","ニ","ヌ","ネ","ノ","ハ","ヒ","フ","ヘ","ホ","マ","ミ","ム","メ","モ","ヤ","ユ","ヨ","ラ","リ","ル","レ","ロ","ワ","ヲ","ン","ガ","ギ","グ","ゲ","ゴ","ザ","ジ","ズ","ゼ","ゾ","ダ","ヂ","ヅ","デ","ド","バ","パ","ビ","ピ","ブ","プ","ベ","ペ","ボ","ポ"]

kanji=["粗","億","達","督","宿","雲","覆","煩","岡","勧","板","糧","熱","思","仏","蔑","哀","朽","貢","凡","実","途","廃","朱","華","舞","計","寿","膜","壊","始","河","軸","策","瞬","貨","止","勝","如","腸","太","後","臆","液","路","舎","緑","松","三","謝","四","払","架","各","内","仁","可","野","武","微","装","揮","皇","害","品","索","審","庁","奉","臣","郷","難","茶","享","意","力","逸","火","慮","応","程","道","似","領","航","乾","溝","協","清","新","既","禁","岩","酷","蔵","嘆","硝","細","盟","橘","移","糾","戦","虫","角","米","抱","侵","美","均","乏","察","色","定","今","炭","話","府","疎","報","出","村","無","等","不","狩","費","薫","隻","画","統","衰","含","規","承","柔","与","漂","丘","輪","霜","製","裕","比","冥","詳","胆","異","烈","須","段","普","効","整","習","張","和","平","管","株","庸","昧","育","鳥","酬","斥","緒","是","曜","掌","遠","神","場","傷","非","雨","界","囲","職","金","訪","暖","釈","台","座","円","守","構","鶴","価","省","扱","宍","性","脈","擁","脳","浄","総","免","顕","位","巨","手","災","君","籍","需","脊","負","朴","探","鼓","慣","予","惑","涙","基","唐","校","精","洗","梅","着","媒","楽","看","警","到","沈","盾","敢","原","隔","月","従","縁","括","玉","預","築","宣","病","引","侮","少","恥","隠","卓","碁","長","尊","旧","質","自","仲","件","勘","拶","市","殺","析","数","磨","択","顧","憤","本","井","危","重","及","彼","充","傑","漁","騒","筋","外","故","違","置","据","苦","失","刻","縮","振","満","慎","床","隆","仮","貴","盆","動","牲","敗","郎","務","吉","希","触","接","洪","口","除","若","犠","義","宮","西","六","曹","冗","歯","組","素","屋","棄","鎮","顔","窮","徒","旨","久","街","衆","皮","判","駅","委","世","問","綻","透","披","斜","納","支","入","責","功","斬","賃","握","堅","欺","併","疑","憂","恩","写","換","威","星","春","燥","遷","迷","人","雰","業","逐","頑","農","渉","期","閉","使","舟","束","励","訓","措","銅","敷","折","枯","遺","待","球","照","符","憎","士","硫","鎌","安","寛","康","変","序","崇","日","了","要","腐","薄","告","子","我","裁","図","越","洋","何","蓋","社","繋","専","地","極","有","欠","幻","忠","錯","茎","綱","衣","受","眠","里","閣","全","区","宗","再","事","血","垂","憲","掲","崩","殖","射","械","壌","検","会","悩","偏","擦","抵","著","酔","語","秘","館","親","愛","涼","植","命","慨","継","経","輸","花","塩","体","挨","休","善","邑","寄","緩","豪","服","京","遭","連","穀","汽","働","甘","秋","由","共","波","熟","型","仕","妖","州","潜","対","中","北","調","室","雄","忌","横","操","治","冷","発","避","寒","必","紀","齢","像","衛","然","活","帰","尋","軌","養","保","城","鎖","身","介","倒","魅","海","給","合","突","戯","織","天","略","傾","買","襲","算","通","砂","知","流","奮","濫","限","銀","討","先","相","低","悪","面","潔","進","時","未","授","賦","境","栽","名","以","露","獲","紡","電","遇","羽","五","川","拙","浴","員","妙","所","周","貫","当","焦","放","追","儀","挑","猶","税","建","節","議","湖","来","光","類","剣","疫","砕","謀","琶","憾","糸","導","物","率","脅","次","宜","想","倣","種","綿","荒","泌","肝","題","死","副","森","土","踏","環","喪","代","蛮","硬","拝","田","弾","説","逆","裸","末","工","山","諏","聴","暗","特","暦","足","祖","強","裏","水","午","識","渡","浦","在","複","染","恵","弊","浮","余","永","司","札","絵","辛","点","愚","妨","感","核","配","転","朝","唆","開","虐","妥","幽","証","起","財","奨","万","匿","条","深","労","欧","赴","最","改","典","約","唯","園","因","勾","多","為","挿","翁","伝","援","軽","官","常","任","鮮","歓","単","淡","没","概","帯","情","戚","渋","成","右","候","級","灰","随","救","懐","船","陽","書","往","資","骨","小","左","交","暴","大","契","乱","挙","商","首","脱","慌","塊","査","風","愁","丁","楯","琵","格","濯","高","矢","抹","派","濃","部","阻","形","留","漫","悠","分","底","党","捨","列","占","送","奇","匠","催","奥","干","浸","献","撃","倍","真","元","復","回","歴","答","潤","壁","究","稲","両","陶","請","化","還","正","鉄","表","凝","忘","切","様","罵","下","滞","準","教","戒","革","寂","遍","緊","却","冶","慢","第","梨","即","厚","遊","係","刺","銘","腺","称","奪","施","毒","至","鉛","被","王","増","徳","黒","加","視","嗣","公","広","眺","雪","徴","翻","怠","累","崎","例","赦","弱","紫","興","厳","遮","提","庫","酵","弄","頂","拘","反","県","独","就","勢","浜","促","港","監","博","漠","酪","邪","技","処","縫","衝","象","拓","軍","備","潟","困","示","陥","敵","誘","科","絡","域","易","摘","枢","密","揺","浪","更","鋭","脂","墨","嫌","抗","考","辞","跡","心","鉱","維","見","者","古","民","助","激","集","泊","震","亜","閑","洲","昇","巧","摩","消","果","層","魂","胞","私","編","主","参","用","兵","痛","現","短","草","量","否","式","季","携","別","律","腎","勤","石","秩","祥","香","互","剰","練","啓","団","優","頻","遂","気","族","雑","霊","念","氾","積","争","態","岸","林","塗","訴","障","八","学","致","藻","征","喚","畑","関","毛","直","況","利","幾","政","償","木","驚","料","銭","唱","孤","執","敏","糖","紛","悲","玄","案","年","燃","行","持","的","絹","依","談","誇","獄","屈","繊","選","停","椎","家","縛","叙","呈","詰","駆","温","範","将","際","棚","白","険","矛","飛","散","絶","惨","抽","印","空","担","脚","潮","飾","頭","湧","秀","奴","住","牧","拡","落","栄","沢","弁","記","静","央","赤","権","能","帳","枚","線","暇","虚","東","都","貿","鈍","扶","彩","吹","逃","拒","夢","黄","捕","沿","終","貌","陸","誕","超","渇","器","明","模","認","臨","軟","退","瞭","膨","枝","閥","蓄","師","立","上","摂","替","理","好","制","伏","堤","抑","較","産","江","刀","賀","癖","生","漢","把","断","拠","走","歌","徹","南","刷","字","前","躍","丹","堆","十","門","富","状","投","令","健","快","佐","紹","史","疾","耕","憶","窒","諸","院","展","穏","己","繁","迫","七","護","源","志","帝","度","階","防","寺","岐","早","結","覚","章","講","醸","曲","席","創","枠","離","冒","圧","値","衡","葉","邦","皆","聞","琴","儒","寧","機","方","辱","国","論","酸","号","述","九","食","陰","同","運","間","沖","攻","瓦","狭","百","聖","島","減","湿","油","鐘","罪","半","髄","割","揚","一","造","裂","網","陳","趣","弧","端","雇","漆","存","求","取","汚","祉","孝","蔽","妄","破","信","融","延","迎","麦","克","布","英","二","盛","培","影","混","靴","容","賠","惰","羅","循","畜","排","窓","曽","得","疲","緯","適","尾","謙","法","仰","懸","根","解","偶","肯","滅","済","指","卑","郭","洞","匹","便","営","確","材","堕","固","恒","藤","湾","泥","過","甲","該","企","韓","御"]

small=["ぁ","ぃ","ぅ","ぇ","ぉ","っ","ゃ","ゅ","ょ","ゎ","ァ","ィ","ゥ","ェ","ォ","ッ","ャ","ュ","ョ","ヮ"]

special=["ー","."]

#combination for handle posible similar and small

filter_word = \
[
  ('レ', 'し', katakana, hiragana),
  ('し', 'レ', hiragana, katakana),
  ('へ', 'ヘ', hiragana, katakana),
  ('ヘ', 'へ', katakana, hiragana),
  ('ベ', 'べ', katakana, hiragana),
  ('べ', 'ベ', hiragana, katakana),
  ('ー', '一', special, kanji),
  ('リ', 'り', katakana, hiragana),
  ('り', 'リ', hiragana, katakana),
  ('ァ', 'ア', small, kanji),
  ('ィ', 'イ', small, kanji),
  ('ゥ', 'ウ', small, kanji),
  ('ェ', 'エ', small, kanji),
  ('ャ', 'ヤ', small, kanji),
  ('ュ', 'ユ', small, kanji),
  ('ョ', 'ヨ', small, kanji),
  ('ッ', 'ツ', small, kanji),
  ('ぁ', 'あ', small, kanji),
  ('ぃ', 'い', small, kanji),
  ('ぅ', 'う', small, kanji),
  ('ぇ', 'え', small, kanji),
  ('ぉ', 'お', small, kanji),
  ('っ', 'つ', small, kanji),
]
  #('つ', 'フ', hiragana, katakana),
  #('フ', 'つ', katakana, hiragana),
  #('う', 'ラ', hiragana, katakana),
  #('ラ', 'う', katakana, hiragana),
  #('こ', 'ニ', hiragana, katakana),
  #('ニ', 'こ', katakana, hiragana),





def choose_Maxheight(text):
    key=1
    for count in range(len(text)):
        if text[count] == "ロ" and len(text)>1:
            key = 2
    return key

#check small character availabilty and handle it 
def check_small(text,height,area,flag):
    try:
        key =choose_Maxheight(text)
        max_height=max(height)
        #key=1
        print(height,area)
        for count in range(len(text)):
            # print(count)
            if flag=="True0":
                if area[count] <= sorted(area)[key] and height[count] <=max_height*0.70:
                    small_key= list(labels.labels.keys())[list(labels.labels.values()).index(text[count])]
                    text[count] =labels1.labels[small_key]

            elif flag =="True1":
                if area[count] <= sorted(area)[key+1] and height[count] <= max_height*0.70:
                    small_key= list(labels.labels.keys())[list(labels.labels.values()).index(text[count])]
                    text[count] =labels1.labels[small_key]

            else:
                if area[count] <= sorted(area)[key-1] and height[count] <= max_height*0.60:
                    small_key= list(labels.labels.keys())[list(labels.labels.values()).index(text[count])]
                    text[count] =labels1.labels[small_key]
    except ValueError as e:
        pass
    return text

#use above combination and handel similar and small cahracter in word
def filter_block(sent):
    
    for i in range(len(sent)):
        for j in range(len(filter_word)):
            if filter_word[j][0] == sent[i]:
                if i==0:
                    bef = filter_word[j][2] is None or (i==0 and sent[i] in filter_word[j][2])
                    aft = filter_word[j][3] is None or (i<len(sent)-1 and sent[i+1] in filter_word[j][3])
                elif i==len(sent)-1:

                    bef = filter_word[j][2] is None or (i==len(sent)-1 and sent[i] in filter_word[j][2])
                    aft = filter_word[j][2] is None or (i>0 and sent[i-1] in filter_word[j][3])
                else:
                    bef = filter_word[j][2] is None or (i>0 and sent[i-1] in filter_word[j][3])
                    aft = filter_word[j][3] is None or (i<len(sent)-1 and sent[i+1] in filter_word[j][3])
                
                if i==0 or i==len(sent)-1:
                    if aft and bef:
                        sent[i] = filter_word[j][1]
                else:
                    if sent[i-1] in special:
                        if aft or bef:
                            sent[i] = filter_word[j][1]
                    else:
                        if aft or bef:
                            sent[i] = filter_word[j][1]

    text = ''.join(str(char) for char in sent)
    return text

# ('り', 'リ', hiragana, katakana),
#   
# ('へ', 'ヘ', hiragana, katakana),
#   ('ヘ', 'へ', katakana, hiragana),
#   ('口', 'ロ', kanji, katakana),
#   ('ロ', '口', katakana, kanji),
#   ('三', 'ミ', kanji, katakana),
#   ('ミ', '三', katakana, kanji),
#   ('十', 'ナ', kanji, katakana),
#   ('ナ', '十', katakana, kanji),
#   ('木', 'ホ', kanji, katakana),
#   ('ホ', '木', katakana, kanji),
#   ('ニ', '二', katakana, kanji),
#   ('二', 'ニ', kanji, katakana)
# sent[i+1] in kanji or 
