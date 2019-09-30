function creatTable(data) {
            //这个函数的参数可以是从后台传过来的也可以是从其他任何地方传过来的
            //这里我假设这个data是一个长度为5的字符串数组 我要把他放在表格的一行里面，分成五列
            var tableData = "<tr>"
            //动态增加5个td,并且把data数组的五个值赋给每个td
            for (var i = 0; i < data.length; i++) {
                tableData += "<td>" + data[i] + "</td>"
            }
            tableData += "</tr>"
            //现在tableData已经生成好了，把他赋值给上面的tbody
            $("#tbody1").html(tableData)
            $("#tbody1").append(tableData);
        }