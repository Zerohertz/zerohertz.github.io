










<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" lang="ko" xml:lang="ko">




<html>

<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">

<head>
<meta http-equiv="X-UA-Compatible" content="IE=edge">
<meta http-equiv="Content-Type" content="text/html;charset=utf-8" />
    <link rel="shortcut icon" href="/LSW/images/ico_favorites.ico"  type="image/x-icon" />
    <link rel="stylesheet" type="text/css" href="/LSW/css/lsw/common.css" />
    <link rel="stylesheet" type="text/css" href="/LSW/js/jquery/jquery.loadmask.css" />
    <!-- 레이아웃 관련 CSS -->
    <link rel="stylesheet" type="text/css" href="/LSW/js/jquery/layout-default-latest.css" />
    <link type="text/css" rel="stylesheet" href="/LSW/css/lsw/layout.css" />
    <link type="text/css" rel="stylesheet" href="/LSW/css/lsw/board.css" />
    <link type="text/css" rel="stylesheet" href="/LSW/css/ui_2017.css" />
    <!-- 레이아웃 관련 CSS //-->

    <script type="text/javascript" src="/LSW/js/jquery/jquery.js"></script>
	<script type="text/javascript" src="/LSW/js/common/common.js"></script>
	<script type="text/javascript" src="/LSW/js/common/drag_search.js"></script>
	<script type="text/javascript" src="/LSW/js/common/CookieUtil.js"></script>
    <script type="text/javascript" src="/LSW/js/jquery/jquery.loadmask.js"></script>
    <script type="text/javascript" src="/LSW/js/common/jquery-custom.js"></script>
    <!-- 레이아웃 관련 JS -->
	<script type="text/javascript" src="/LSW/js/jquery/jquery-ui.js"></script>
	<script type="text/javascript" src="/LSW/js/jquery/jquery-ui_2017.js"></script>
	<script type="text/javascript" src="/LSW/js/jquery/jquery.layout-latest.js"></script>
	<script type="text/javascript" src="/LSW/js/common/layout.js"></script>
	<!-- 레이아웃 관련 JS //-->
	
    <style type="text/css">
    .ui-layout-pane {
		padding: 0;
		background:	#EEE;
	}
	.ui-layout-west {
		background:	#CFC;
	}
	.ui-layout-center {
		background:	#FFC;
		padding:	0; /* IMPORTANT - remove padding so pane can 'collapse' to 0-width */
	}

	.ui-layout-west > .ui-layout-center {
		background:	#CFC;
	}

	.ui-layout-west > .ui-layout-south {
		background:	#AFE;
	}
	.ui-layout-pane { border-width: 0; }
	.ui-layout-north { border-bottom-width:	1px; }
	.ui-layout-south { border-top-width: 1px; }
	.ui-layout-resizer-west {
		border-width: 0;
		border-left: 2px solid #b4b4b4;
		background-color:white;
	}
	.ui-layout-toggler {
		border: 0;
		background-color:#fff
	}
	.ui-layout-toggler-west-closed {
		background-image: url("./images/button/leslidev7_2.gif");
		background-color:white;
		background-repeat: no-repeat;
	}
	.ui-layout-toggler-west-open {
		background-image: url("./images/button/leslidev7_1.gif");
     	background-color:white;
     	background-repeat: no-repeat;
	}
    </style>
    
    <script type="text/javascript">
        var webRoot = '/LSW';
    </script>
</head>

<body>
    










<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">

<html xmlns="http://www.w3.org/1999/xhtml">



<head>
	<link rel="stylesheet" type="text/css" href="/LSW/css/lsw/content.css"/>
    <title>국가법령정보센터 | 오류페이지</title>
    <script type="text/javascript">
		/*#13176 전화 번호 일주일단위 교차출력*/
		$(document).ready(function(){
			var preDay = '2019-02-11';
			var toDay = new Date();
		
			preDay = new Date(preDay);
			diffPreDay = new Date(preDay.getFullYear(), preDay.getMonth()+1, preDay.getDate());
			diffToDay = new Date(toDay.getFullYear(), toDay.getMonth()+1, toDay.getDate());
		
			var diff = Math.abs(diffToDay.getTime() - diffPreDay.getTime());
			diff = Math.ceil(diff / (1000*3600*24));
			var flag = Math.floor((diff / 7) % 2);
			
			if(flag == 0){
				$('#lawTelNum').html("1551-3060 1번");
			}else if(flag == 1){
				$('#lawTelNum').html("1551-3060 1번");
			}
		});
    </script>
</head>

<body> 

<div id="error500">
	<h1><a href="javascript:;"><img src="/LSW/images/main/h1_main1.gif" alt="신뢰할 수 있는 법제처. 국가법령정보센터"/></a></h1>
	
	<div class="error_txt">
		
		
			
			
				<h2>서비스 이용에 불편을 드려서 죄송합니다</h2>
				<p>현재 사용자가 많아 요청하신 페이지를 정상적으로 제공할 수 없습니다. <br/>잠시 후 다시 접속해 주시기 바랍니다.<br/>동일한 문제가 지속될 경우 아래 번호로 문의해주시기 바랍니다.</p>
				<em>법제처 법령데이터혁신팀 ( <span id="lawTelNum">1551-3060 1번</span> )</em>
			
		
	</div>
</div><!-- error500 -->
 
</body>
</html>
    <script type="text/javascript" src="/LSW/js/common/hrefRemove.js"></script>
</body>
</html>