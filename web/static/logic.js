function pad(num){
    return ("000000" + num).substr(-6,6)
}

function getAbsOffset(el) {
    const rect = el.getBoundingClientRect();
        return {
            left: rect.left + window.scrollX,
            top: rect.top + window.scrollY
        };
}

angular
.module('AdvApp', ['infinite-scroll'])
.controller('clevrImagesController', ['$scope', '$http', '$timeout', '$sce', function($scope, $http, $timeout, $sce){
    var thiz = this;
    var queryChoice = -1;
    var questionChoice = -1;
    thiz.loadMoreDisabled = false;
    var toLoad = 8; 
    var maxLoadedImages = 48;
    thiz.data = [];
    thiz.loadedData = [];
    thiz.queryChoicePadded = queryChoice;
    var head = 0;
    var initialQueries = [2,4,7,9,14,15,16,18,19,21,22,23,25,26,33,35,37,38,40,42,44,45,47,48,50,51,52,55,56];
    var nextStop = maxLoadedImages;
    var selectedQuestion = "";
    var recognition = null;

    //adjustAffixWidth();

    /*$http.get('http://datone.isti.cnr.it/adversarials/adv_scores.json').then(function(response){
        thiz.data = $filter('orderBy')(response.data, 'knn_score', true);
        thiz.loadedData = thiz.data.slice(0, 20);
    });*/
    $scope.changeQueryDivStyle = {'display': 'none'};
    $scope.chooseQueryDivStyle = {'display': 'block'};
    $scope.template = 'templateChooseImage.html';
    $scope.translate = true;
                            
    function load() {
        if(queryChoice < 0){
            for(var i = 0; i < toLoad; i++) {
                var curIndex = (i + head<initialQueries.length) ? initialQueries[i + head] : (57 + i + head);
                thiz.loadedData.push([curIndex, pad(curIndex)]);
            }
        } else {
            for(var i = 0; i < toLoad; i++) {
                var curIndex = thiz.data[i + head];
                thiz.loadedData.push([curIndex, pad(curIndex)]);
            }
        }
        head += toLoad;
    }

    function startRecognition() {
        recognition = new webkitSpeechRecognition();
        recognition.lang = ($scope.translate) ? "it" : "en";
        recognition.continuous = true;
        recognition.onresult = function (event) {
            var transcripted = "";
            for (var i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    console.log(event.results[i][0].transcript);
                    console.log('###-');
                    transcripted = transcripted.concat(event.results[i][0].transcript);
                }
            }
            transcripted = transcripted.concat('?');
            $scope.sendQuestion(transcripted);
        };
        recognition.onerror = function(event) {
            $scope.$apply(function(){
                $scope.recognizeOn = false;
            });
            console.log('Speech recognition error');
        };

        recognition.onend = function() {
            $scope.$apply(function(){
                $scope.recognizeOn = false;
            });
            console.log('Speech recognition service disconnected');
        }

        recognition.start();
        $scope.recognizeOn = true;
    }

    function stopRecognition() {
        if (recognition!=null) recognition.stop();
        recognition = null;
    }

    load();

    $scope.startRecognition = function(){
        startRecognition();
    }

    $scope.linkDisabled = function() {
        return queryChoice>=0;
    }
    $scope.changePaddingClass = function() {
        if (queryChoice>=0) {
            return 'more-padding';
        } else {
            return 'less-padding';
        }
    }
    $scope.setAffixOffset = function() {
        var init_off = 625;
        var off = getAbsOffset($( ".samples-container" )[ 0 ]);
        if (queryChoice>=0) {
            return 750 + 0.8*(off.top - init_off);
        } else {
            return 510 + 0.8*(off.top - init_off);
        }
    }

    $scope.queryChoice = function() {
        $scope.template = 'templateChooseImage.html';
        stopRecognition();
        queryChoice = -1; 
        thiz.loadedData = [];
        nextStop = maxLoadedImages;
        thiz.loadMoreDisabled = false;
        head = 0;
        $scope.changeQueryDivStyle = {'display': 'none'};
        $scope.chooseQueryDivStyle = {'display': 'block'};
        load();
    }    

    $scope.loadMore = function() {
        nextStop += maxLoadedImages;
        thiz.loadMoreDisabled = false; 
        load();
    }

    $scope.chosenQuery = function(id) {
        $scope.answerDivStyle = {'display': 'none'};
        thiz.loadMoreDisabled = true;
        thiz.loadedData = [];
        queryChoice = id;
        thiz.queryChoicePadded = pad(queryChoice);
        $scope.changeQueryDivStyle = {};
        $scope.chooseQueryDivStyle = {'display': 'none'};
        $scope.template = 'templateChooseQuestion.html';

        $http({
            url: '/requests/questions',
            method: 'GET',
            params: {imgid: id}}).then(function(response){
                var rawquestions = response.data;
                thiz.questions = [];
                //translate if necessary
                if ($scope.translate){
                    for (i=0; i<rawquestions.length; i++){
                        (function(inti){
                            rawquestions[inti].sentence = rawquestions[inti].sentence.replace('left of', 'to the left of');
                            $scope.translateQuestion(rawquestions[inti].sentence, function(data){
                                rawquestions[inti].sentence = data.translatedText;
                            }, 'en', 'it');
                        })(i);
                    }
                }
                thiz.questions = rawquestions;
                questionChoice = response.data[0].id;  //the first question on this image 

                //start speech recognition
                startRecognition();     
            
            /*head = 0;
            nextStop = maxLoadedImages;
            load();
            thiz.loadMoreDisabled = false;*/
            
        });
        
    };

    $scope.fillQuestionInput = function(qst_id) {
        var qst = "";
        //find the question having qst_id. TODO: actually, inefficient implementation
        for (var i=0; i<thiz.questions.length; i++){
            if (thiz.questions[i].id == qst_id){
                qst = thiz.questions[i].sentence;
            }
        }
        $scope.inputValue = qst;
        selectedQuestion = qst;
        questionChoice = qst_id;
        $scope.textual=true;
        $scope.$broadcast('questionSelected');
    }

    $scope.inputValue = null;
    $scope.sendQuestion = function(qst) {
        if (selectedQuestion == qst){
            //the query was not modified by the user
            params = {qstid: questionChoice};
            is_id = true;
        } else {
            qst = qst.replace('.',';');
            params = {sentence: qst, qstid: questionChoice};
            is_id = false;
        }
        
        var makeQueryRequest = function(params){ 
            $scope.loadingAnswer = true;
            $http({
                url: '/requests/answer',
                method: 'GET',
                params: params}). then(function(response){
                    $scope.answerReady = true;
                    $scope.answerDivStyle = {'width': '90%'};
                    if (response.data[2] == 'error'){
                        thiz.answer = ($scope.translate) ? 'Mi dispiace, non ho capito la domanda' : "I'm sorry, cannot understand the question";
                    } else {
                        if ($scope.translate){
                            thiz.answer = $scope.translateAnswer(response.data[0]);
                        } else {
                            thiz.answer = response.data[0];
                        }
                        /*if (is_id){  
                            if (response.data[0] == response.data[1]) {
                                $scope.answerDivStyle = {'color': 'green'}; 
                            } else {
                                $scope.answerDivStyle = {'color': 'red'};
                            }
                        }*/
                    }
                $timeout(function(){
                    $scope.answerDivStyle = {};
                    $scope.answerReady = false;
                    $scope.loadingAnswer = false;
                }, 2000);
                /*head = 0;
                nextStop = maxLoadedImages;
                load();
                thiz.loadMoreDisabled = false;*/
                
            });
        }

        if ($scope.translate && !is_id){ 
            $scope.translateQuestion(params.sentence, function(data){
                params.sentence = data.translatedText;
                makeQueryRequest(params);
            }, 'it', 'en');
        } else {
            makeQueryRequest(params);
        }
          
    }

    $scope.translateQuestion = function(text,callback,sourceLang,targetLang) {
        var translateAPIURL = $sce.trustAsResourceUrl('https://script.google.com/macros/s/AKfycbwaJoJiJdAEgYLRgBkFoPo2tx9HhM6PLM_y6fWNfHVIMy7_JMIk/exec')
        //make a jsonp request to the translate webapp
        $http.jsonp(
            translateAPIURL,
            {jsonpCallbackParam: 'callback', params:{q: text, source:sourceLang, target:targetLang}}
        ).then(function(data){
            callback(data.data);
        });
    }

    /*$scope.translateQuestion = function(text,callback,sourceLang,targetLang) {
        $http({
                url: 'https://script.google.com/macros/s/AKfycbwaJoJiJdAEgYLRgBkFoPo2tx9HhM6PLM_y6fWNfHVIMy7_JMIk/exec',
                method: 'GET',
                params:{q: text, source:sourceLang, target:targetLang}}
        ).then(function(data){
            callback(data.data);
        });
    }*/

    $scope.translateAnswer = function(answer) {
        var dict = {
            large: 'grande',
            small: 'piccolo',
            rubber: 'di gomma',
            metal: 'di metallo',
            gray: 'grigio',
            blue: 'blu',
            brown: 'marrone',
            yellow: 'giallo',
            purple: 'viola',
            green: 'verde',
            cyan: 'azzurro',
            red: 'rosso',
            sphere: 'una sfera',
            cube: 'un cubo',
            cylinder: 'un cilindro',
            yes: 'sì',
            no: 'no' 
        };

        if (answer in dict){
            return dict[answer];
        } else {
            return answer;
        }
    }

    $scope.changeInputType = function(isTextual){
        if (isTextual) {
            stopRecognition();
        } else {
            startRecognition();
        }
    }

    thiz.loadMore = function() {
        if (head > nextStop){
            thiz.loadMoreDisabled = true;
            return;
        }
        if (thiz.loadMoreDisabled){
            //spurious triggers due to concurrency
            return;
        }
        
        load();
    };
}])
.directive('imgFadeInOnload', function () {
    return {
      restrict: 'A',
      link: function postLink(scope, element, attr) {
        element.css('opacity', 0);
        element.css('-moz-transition', 'opacity 2s');
        element.css('-webkit-transition', 'opacity 2s');
        element.css('-o-transition', 'opacity 1s');
        element.css('transition', 'opacity 1s');
        element.bind("load", function () {
          element.css('opacity', 1);
        });
        element.attr('src', attr.imgFadeInOnload);
      }
    };
})
.directive('focusOn', function() {
   return function(scope, elem, attr) {
      scope.$on(attr.focusOn, function(e) {
          elem[0].focus();
      });
   };
});
