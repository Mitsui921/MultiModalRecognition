// $(function(){
//   var n=0;
//   var res =0;
//   var myArray=new Array()
//   let peoples = [
//     { number: '1', w: 1 },
//     { number: '2', w: 4 },
//     { number: '3', w: 6 },
//     { number: '4', w: 9 },
//     { number: '5', w: 12 },
//     { number: '6', w: 18 },
//     { number: '7', w: 50 },
//   ];
//   var test = "http://127.0.0.1:80/rw/iosystem/signals/do_p_j1?action=set"; // 需要提交的变量
//   document.getElementById("set1").value = test;
  
// })
$(function(){
  // var Ip_Port="127.0.0.1:80";
  var Ip_Port = "51r7m51573.qicp.vip";


  $("#btnZ2").click(function(){
    $.ajax({
      url:"http://"+Ip_Port+"/rw/iosystem/signals/do_z2?action=set",//要请求的服务器url
      data:{"lvalue":1},// 
      contentType : "application/x-www-form-urlencoded; charset=UTF-8",
      async:true,//是否是异步请求
      cache:false,//是否缓存结果
      type:"Post",//请求方式
      dataType:"xml",//服务器返回什么类型数据 text xml javascript json(javascript对象)
      success:function(result){//函数会在服务器执行成功后执行，result就是服务器返回结果

      },
      error:function(jqXHR, textStatus, errorThrown) 
      {
          console.log(result);
      }
    });
  })
  $("#btnZ1").click(function(){
    $.ajax({
      url:"http://"+Ip_Port+"/rw/iosystem/signals/do_z1?action=set",//要请求的服务器url
      data:{"lvalue":1},// 
      contentType : "application/x-www-form-urlencoded; charset=UTF-8",
      async:true,//是否是异步请求
      cache:false,//是否缓存结果
      type:"Post",//请求方式
      dataType:"xml",//服务器返回什么类型数据 text xml javascript json(javascript对象)
      success:function(result){//函数会在服务器执行成功后执行，result就是服务器返回结果

      },
      error:function(jqXHR, textStatus, errorThrown) 
      {
          console.log(result);
      }
    });
  })
  $("#btnY2").click(function(){
    $.ajax({
      url:"http://"+Ip_Port+"/rw/iosystem/signals/do_y2?action=set",//要请求的服务器url
      data:{"lvalue":1},// 
      contentType : "application/x-www-form-urlencoded; charset=UTF-8",
      async:true,//是否是异步请求
      cache:false,//是否缓存结果
      type:"Post",//请求方式
      dataType:"xml",//服务器返回什么类型数据 text xml javascript json(javascript对象)
      success:function(result){//函数会在服务器执行成功后执行，result就是服务器返回结果

      },
      error:function(jqXHR, textStatus, errorThrown) 
      {
          console.log(result);
      }
    });
  })
  $("#btnY1").click(function(){
    $.ajax({
      url:"http://"+Ip_Port+"/rw/iosystem/signals/do_y1?action=set",//要请求的服务器url
      data:{"lvalue":1},// 
      contentType : "application/x-www-form-urlencoded; charset=UTF-8",
      async:true,//是否是异步请求
      cache:false,//是否缓存结果
      type:"Post",//请求方式
      dataType:"xml",//服务器返回什么类型数据 text xml javascript json(javascript对象)
      success:function(result){//函数会在服务器执行成功后执行，result就是服务器返回结果

      },
      error:function(jqXHR, textStatus, errorThrown) 
      {
          console.log(result);
      }
    });
  })
  $("#btnX2").click(function(){
    $.ajax({
      url:"http://"+Ip_Port+"/rw/iosystem/signals/do_x2?action=set",//要请求的服务器url
      data:{"lvalue":1},// 
      contentType : "application/x-www-form-urlencoded; charset=UTF-8",
      async:true,//是否是异步请求
      cache:false,//是否缓存结果
      type:"Post",//请求方式
      dataType:"xml",//服务器返回什么类型数据 text xml javascript json(javascript对象)
      success:function(result){//函数会在服务器执行成功后执行，result就是服务器返回结果

      },
      error:function(jqXHR, textStatus, errorThrown) 
      {
          console.log(result);
      }
    });
  })
  $("#btnX1").click(function(){
    $.ajax({
      url:"http://"+Ip_Port+"/rw/iosystem/signals/do_x1?action=set",//要请求的服务器url
      data:{"lvalue":1},// 
      contentType : "application/x-www-form-urlencoded; charset=UTF-8",
      async:true,//是否是异步请求
      cache:false,//是否缓存结果
      type:"Post",//请求方式
      dataType:"xml",//服务器返回什么类型数据 text xml javascript json(javascript对象)
      success:function(result){//函数会在服务器执行成功后执行，result就是服务器返回结果

      },
      error:function(jqXHR, textStatus, errorThrown) 
      {
          console.log(result);
      }
    });
  })
  $("#btn_jog6_1").click(function(){
    $.ajax({
      url:"http://"+Ip_Port+"/rw/iosystem/signals/do_p_j6?action=set",//要请求的服务器url
      data:{"lvalue":1},// 
      contentType : "application/x-www-form-urlencoded; charset=UTF-8",
      async:true,//是否是异步请求
      cache:false,//是否缓存结果
      type:"Post",//请求方式
      dataType:"xml",//服务器返回什么类型数据 text xml javascript json(javascript对象)
      success:function(result){//函数会在服务器执行成功后执行，result就是服务器返回结果

      },
      error:function(jqXHR, textStatus, errorThrown) 
      {
          console.log(result);
      }
    });
  })
  $("#btn_jog6_2").click(function(){
    $.ajax({
      url:"http://"+Ip_Port+"/rw/iosystem/signals/do_n_j6?action=set",//要请求的服务器url
      data:{"lvalue":1},// 
      contentType : "application/x-www-form-urlencoded; charset=UTF-8",
      async:true,//是否是异步请求
      cache:false,//是否缓存结果
      type:"Post",//请求方式
      dataType:"xml",//服务器返回什么类型数据 text xml javascript json(javascript对象)
      success:function(result){//函数会在服务器执行成功后执行，result就是服务器返回结果

      },
      error:function(jqXHR, textStatus, errorThrown) 
      {
          console.log(result);
      }
    });
  })
  $("#btn_jog5_1").click(function(){
    $.ajax({
      url:"http://"+Ip_Port+"/rw/iosystem/signals/do_p_j5?action=set",//要请求的服务器url
      data:{"lvalue":1},// 
      contentType : "application/x-www-form-urlencoded; charset=UTF-8",
      async:true,//是否是异步请求
      cache:false,//是否缓存结果
      type:"Post",//请求方式
      dataType:"xml",//服务器返回什么类型数据 text xml javascript json(javascript对象)
      success:function(result){//函数会在服务器执行成功后执行，result就是服务器返回结果

      },
      error:function(jqXHR, textStatus, errorThrown) 
      {
          console.log(result);
      }
    });
  })
  $("#btn_jog5_2").click(function(){
    $.ajax({
      url:"http://"+Ip_Port+"/rw/iosystem/signals/do_n_j5?action=set",//要请求的服务器url
      data:{"lvalue":1},// 
      contentType : "application/x-www-form-urlencoded; charset=UTF-8",
      async:true,//是否是异步请求
      cache:false,//是否缓存结果
      type:"Post",//请求方式
      dataType:"xml",//服务器返回什么类型数据 text xml javascript json(javascript对象)
      success:function(result){//函数会在服务器执行成功后执行，result就是服务器返回结果

      },
      error:function(jqXHR, textStatus, errorThrown) 
      {
          console.log(result);
      }
    });
  })
  $("#btn_jog4_1").click(function(){
    $.ajax({
      url:"http://"+Ip_Port+"/rw/iosystem/signals/do_p_j4?action=set",//要请求的服务器url
      data:{"lvalue":1},// 
      contentType : "application/x-www-form-urlencoded; charset=UTF-8",
      async:true,//是否是异步请求
      cache:false,//是否缓存结果
      type:"Post",//请求方式
      dataType:"xml",//服务器返回什么类型数据 text xml javascript json(javascript对象)
      success:function(result){//函数会在服务器执行成功后执行，result就是服务器返回结果

      },
      error:function(jqXHR, textStatus, errorThrown) 
      {
          console.log(result);
      }
    });
  })
  $("#btn_jog4_2").click(function(){
    $.ajax({
      url:"http://"+Ip_Port+"/rw/iosystem/signals/do_n_j4?action=set",//要请求的服务器url
      data:{"lvalue":1},// 
      contentType : "application/x-www-form-urlencoded; charset=UTF-8",
      async:true,//是否是异步请求
      cache:false,//是否缓存结果
      type:"Post",//请求方式
      dataType:"xml",//服务器返回什么类型数据 text xml javascript json(javascript对象)
      success:function(result){//函数会在服务器执行成功后执行，result就是服务器返回结果

      },
      error:function(jqXHR, textStatus, errorThrown) 
      {
          console.log(result);
      }
    });
  })
  $("#btn_jog3_1").click(function(){
    $.ajax({
      url:"http://"+Ip_Port+"/rw/iosystem/signals/do_p_j3?action=set",//要请求的服务器url
      data:{"lvalue":1},// 
      contentType : "application/x-www-form-urlencoded; charset=UTF-8",
      async:true,//是否是异步请求
      cache:false,//是否缓存结果
      type:"Post",//请求方式
      dataType:"xml",//服务器返回什么类型数据 text xml javascript json(javascript对象)
      success:function(result){//函数会在服务器执行成功后执行，result就是服务器返回结果

      },
      error:function(jqXHR, textStatus, errorThrown) 
      {
          console.log(result);
      }
    });
  })
  $("#btn_jog3_2").click(function(){
    $.ajax({
      url:"http://"+Ip_Port+"/rw/iosystem/signals/do_n_j3?action=set",//要请求的服务器url
      data:{"lvalue":1},// 
      contentType : "application/x-www-form-urlencoded; charset=UTF-8",
      async:true,//是否是异步请求
      cache:false,//是否缓存结果
      type:"Post",//请求方式
      dataType:"xml",//服务器返回什么类型数据 text xml javascript json(javascript对象)
      success:function(result){//函数会在服务器执行成功后执行，result就是服务器返回结果

      },
      error:function(jqXHR, textStatus, errorThrown) 
      {
          console.log(result);
      }
    });
  })

  $("#btn_jog2_1").click(function(){
    $.ajax({
      url:"http://"+Ip_Port+"/rw/iosystem/signals/do_p_j2?action=set",//要请求的服务器url
      data:{"lvalue":1},// 
      contentType : "application/x-www-form-urlencoded; charset=UTF-8",
      async:true,//是否是异步请求
      cache:false,//是否缓存结果
      type:"Post",//请求方式
      dataType:"xml",//服务器返回什么类型数据 text xml javascript json(javascript对象)
      success:function(result){//函数会在服务器执行成功后执行，result就是服务器返回结果

      },
      error:function(jqXHR, textStatus, errorThrown) 
      {
          console.log(result);
      }
    });
  })
  $("#btn_jog2_2").click(function(){
    $.ajax({
      url:"http://"+Ip_Port+"/rw/iosystem/signals/do_n_j2?action=set",//要请求的服务器url
      data:{"lvalue":1},// 
      contentType : "application/x-www-form-urlencoded; charset=UTF-8",
      async:true,//是否是异步请求
      cache:false,//是否缓存结果
      type:"Post",//请求方式
      dataType:"xml",//服务器返回什么类型数据 text xml javascript json(javascript对象)
      success:function(result){//函数会在服务器执行成功后执行，result就是服务器返回结果

      },
      error:function(jqXHR, textStatus, errorThrown) 
      {
          console.log(result);
      }
    });
  })
  $("#btn_jog1_1").click(function(){
    $.ajax({
      url:"http://"+Ip_Port+"/rw/iosystem/signals/do_p_j1?action=set",//要请求的服务器url
      data:{"lvalue":1},// 
      contentType : "application/x-www-form-urlencoded; charset=UTF-8",
      async:true,//是否是异步请求
      cache:false,//是否缓存结果
      type:"Post",//请求方式
      dataType:"xml",//服务器返回什么类型数据 text xml javascript json(javascript对象)
      success:function(result){//函数会在服务器执行成功后执行，result就是服务器返回结果

      },
      error:function(jqXHR, textStatus, errorThrown) 
      {
          console.log(result);
      }
    });
  })
  $("#btn_jog1_2").click(function(){
    $.ajax({
      url:"http://"+Ip_Port+"/rw/iosystem/signals/do_n_j1?action=set",//要请求的服务器url
      data:{"lvalue":1},// 
      contentType : "application/x-www-form-urlencoded; charset=UTF-8",
      async:true,//是否是异步请求
      cache:false,//是否缓存结果
      type:"Post",//请求方式
      dataType:"xml",//服务器返回什么类型数据 text xml javascript json(javascript对象)
      success:function(result){//函数会在服务器执行成功后执行，result就是服务器返回结果

      },
      error:function(jqXHR, textStatus, errorThrown) 
      {
          console.log(result);
      }
    });
  })
  $("#btn_ppToMain").click(function(){
    $.ajax({
      url:"http://"+Ip_Port+"/rw/rapid/execution?action=resetpp",//要请求的服务器url
      //data:{"lvalue":1},// 
      contentType : "application/x-www-form-urlencoded; charset=UTF-8",
      async:true,//是否是异步请求
      cache:false,//是否缓存结果
      type:"Post",//请求方式
      dataType:"xml",//服务器返回什么类型数据 text xml javascript json(javascript对象)
      success:function(result){//函数会在服务器执行成功后执行，result就是服务器返回结果

      },
      error:function(jqXHR, textStatus, errorThrown) 
      {
          console.log(result);
      }
    });
  })

  $("#btn_Start").click(function(){
    $.ajax({
      url:"http://"+Ip_Port+"/rw/rapid/execution?action=start",//要请求的服务器url
      data:{"regain":"continue","execmode":"continue","cycle":"forever","condition":"none","stopatbp":"disabled","alltaskbytsp":"false"},// 
      contentType : "application/x-www-form-urlencoded; charset=UTF-8",
      async:true,//是否是异步请求
      cache:false,//是否缓存结果
      type:"Post",//请求方式
      dataType:"xml",//服务器返回什么类型数据 text xml javascript json(javascript对象)
      success:function(result){//函数会在服务器执行成功后执行，result就是服务器返回结果

      },
      error:function(jqXHR, textStatus, errorThrown) 
      {
          console.log(result);
      }
    });
  })
  $("#btn_Stop").click(function(){
    $.ajax({
      url:"http://"+Ip_Port+"/rw/rapid/execution?action=stop",//要请求的服务器url
      data:{"stopmode":"stop","usetsp":"normal"},// 
      contentType : "application/x-www-form-urlencoded; charset=UTF-8",
      async:true,//是否是异步请求
      cache:false,//是否缓存结果
      type:"Post",//请求方式
      dataType:"xml",//服务器返回什么类型数据 text xml javascript json(javascript对象)
      success:function(result){//函数会在服务器执行成功后执行，result就是服务器返回结果

      },
      error:function(jqXHR, textStatus, errorThrown) 
      {
          console.log(result);
      }
    });
  })

  $("#ModifyStep").click(function(){
    $.ajax({
      url:"http://"+Ip_Port+"/rw/rapid/symbol/data/RAPID/T_ROB1/user/reg1?action=set",//要请求的服务器url
      data:{"value":input_val.value},// 
      contentType : "application/x-www-form-urlencoded; charset=UTF-8",
      async:true,//是否是异步请求
      cache:false,//是否缓存结果
      type:"Post",//请求方式
      dataType:"xml",//服务器返回什么类型数据 text xml javascript json(javascript对象)
      success:function(result){//函数会在服务器执行成功后执行，result就是服务器返回结果

      },
      error:function(jqXHR, textStatus, errorThrown) 
      {
          console.log(result);
      }
    });
  })
  
})



