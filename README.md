### 答题卡识别-返回答题矩阵
地址: /recog

方法: post

参数
```
col	//行
row	//列
card //图片绝对地址或图片
```

返回
```

{
	"status": 0, 
	"msg": "", 
	"ans": [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
}
```

### 答题卡识别-条码识别
地址: /barcode

方法: post

参数
```
barcode //图片绝对地址或图片
```

返回
```

{
	"status": 0, 
	"msg": "", 
	"ans": 6937748305057
}
```

### 答题卡识别-返回答题分数结果
地址: /markingScore

方法: post

参数
```
col	//行
row	//列
card //图片绝对地址
standAns // 标准答案.A,B|C|D|A
quesType // choice: 2|multiChoice: 3
direction // horizontal: 1|vertical: 2
totalScore // 题块总分
```

返回
```

{
	"status": 0, 
	"msg": "", 
	"ans": [[0, 1, 2, 3], "A,B|C|D|-"]
}
```

### 答题卡识别-识别学号
地址: /studentIdRecog

方法: post

参数
```
code // 条码或填涂区域图片绝对地址
```

返回
```

{
	"status": 0, 
	"msg": "", 
	"ans": 6937748305057
}
```