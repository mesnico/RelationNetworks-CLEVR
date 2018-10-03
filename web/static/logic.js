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

    var recognitionEnd = false;
    function startRecognition() {
        recognition = new webkitSpeechRecognition();
        recognition.lang = ($scope.translate) ? "it" : "en";
        recognition.continuous = false;
        recognition.interimResults = true;
        recognition.onresult = function (event) {
            var transcripted = "";
            $scope.interimText = "";
            if (typeof(event.results) == 'undefined') {
                stopRecognition();
                return;
            }
            for (var i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    console.log(event.results[i][0].transcript);
                    console.log('###-');
                    transcripted = transcripted.concat(event.results[i][0].transcript);
                    transcripted = transcripted.concat('?');
                    $scope.interimText = transcripted;
                    $scope.sendQuestion(transcripted);
                    recognitionEnd = true;
                    //canDeleteInterim = true;
                } else {
                    //if (canDeleteInterim) $scope.interimText = "";
                    $scope.$apply(function(){
                        $scope.interimText += event.results[i][0].transcript;
                    });
                    
                    console.log($scope.interimText);
                }
            }
        };
        recognition.onerror = function(event) {
            $scope.$apply(function(){
                $scope.recognizeOn = false;
            });
            console.log('Speech recognition error');
        };

        recognition.onend = function() {
            $scope.$apply(function(){
                /*if (recognitionEnd){
                    $scope.recognizeOn = false;
                } else {
                    startRecognition();
                }*/
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
        if (!$scope.textual){
            startRecognition();
        }

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
                            //rawquestions[inti].sentence = rawquestions[inti].sentence.replace('left of', 'to the left of');
                            rawquestions[inti].sentence = toUIPreTranslationProcessing(rawquestions[inti].sentence);
                            $scope.translateQuestion(rawquestions[inti].sentence, function(data){
                                rawquestions[inti].sentence = data.translatedText;
                            }, 'en', 'it');
                        })(i);
                    }
                }
                thiz.questions = rawquestions;
                questionChoice = response.data[0].id;  //the first question on this image  
            
            /*head = 0;
            nextStop = maxLoadedImages;
            load();
            thiz.loadMoreDisabled = false;*/
            
        });
        
    };

    $scope.fillQuestionInput = function(qst_id) {
        var qst = "";
        stopRecognition();
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

    function toNetPostTranslationProcessing(sentence){
        var proc;
        proc = sentence.replace('.',';');
        proc = proc.replace('colour','color');
        proc = proc.replace(/which/i,'what');
        proc = proc.replace('opaque','rubber');
        proc = proc.replace(/there\'s/i,'there is');
        proc = proc.replace(/what\'s/i,'what is');
        proc = proc.replace('violet','purple');
        proc = proc.replace(/dimension.?/,'size');
        proc = proc.replace(' bright ',' metal ');
        //proc = proc.replace(/viola/i,'purple');
        proc = proc.replace('besides','other than');
        proc = proc.replace('the one','that');
        proc = proc.replace(/\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b/,'');
        proc = proc.replace(' all ',' ');
        proc = proc.replace(' some ',' ');
        return proc;
    }

    function toNetPreTranslationProcessing(sentence){
        var proc;
        proc = sentence.replace(/azzurr./,'ciano');
        return proc;
    }

    function toUIPreTranslationProcessing(sentence){
        var proc;
        proc = sentence.replace(/(?<!to the )left of/,'to the left of');
        proc = proc.replace(/(?<!to the )right of/,'to the right of');
        return proc;
    } 

    function showAnswer(answer){
        $scope.answerReady = true;
        //$scope.answerDivStyle = {'width': '90%'};
        thiz.answer = answer;
        $timeout(function(){
            $scope.answerDivStyle = {};
            $scope.answerReady = false;
            $scope.loadingAnswer = false;
            if (!$scope.textual){
                recognitionEnd = false;
                startRecognition();
            }
        }, 2000);
    }

    function funnyResponse(question){
        var answer = null;
        question = question.replace('?','');
        var badItWords = ['stronzo','scemo','stupido','bastardo','coglione','fanculo','mamma','ohi','ehi','ei','oi'];
        var chunks = question.split(" ");
        for (var i=0; i<chunks.length; i++){
            if (badItWords.includes(chunks[i])){
                answer = 'Stai tranquillo';
            }
        }

        if (question.search(/\*+/)>=0){
            answer = 'Stai tranquillo';
        }

        if (chunks.includes('ciao')){
            answer = 'Ciao';
        }
        
        if (chunks.includes('mazzoli') || chunks.includes('massoli')){
            answer = 'Massoli... attenzione...';
        }

        if (chunks.includes('bravo')){
            answer = 'grazie';
        }

        if (chunks.includes('siri')){
            answer = 'Ho ucciso Siri';
        } 

        if (chunks.includes('senso') && chunks.includes('vita')){
            answer = '42';
        }

        return answer;
    }

    $scope.inputValue = null;
    $scope.sendQuestion = function(qst) {
        //abort if empty text-box
        if (qst==null || qst.length==0){
            return;
        }
        if (selectedQuestion == qst){
            //the query was not modified by the user
            params = {qstid: questionChoice};
            is_id = true;
        } else {
            params = {sentence: qst, qstid: questionChoice};
            is_id = false;
        }
        
        $scope.loadingAnswer = true;

        var makeQueryRequest = function(params){ 
            var answer;
            $http({
                url: '/requests/answer',
                method: 'GET',
                params: params}). then(function(response){
                    
                    if (response.data[2] == 'error'){
                        answer = ($scope.translate) ? 'Mi dispiace, non ho capito la domanda' : "I'm sorry, cannot understand the question";
                    } else {
                        if ($scope.translate){
                            answer = $scope.translateAnswer(response.data[0]);
                        } else {
                            answer = response.data[0];
                        }
                        /*if (is_id){  
                            if (response.data[0] == response.data[1]) {
                                $scope.answerDivStyle = {'color': 'green'}; 
                            } else {
                                $scope.answerDivStyle = {'color': 'red'};
                            }
                        }*/
                    }
                    showAnswer(answer);
                });
        }

        if ($scope.translate && !is_id){
            var preproc = toNetPreTranslationProcessing(params.sentence);
            preproc = preproc.toLowerCase();
            var funny = funnyResponse(preproc);
            if (funny != null){
                showAnswer(funny);
            } else {
                $scope.translateQuestion(preproc, function(data){
                    params.sentence = toNetPostTranslationProcessing(data.translatedText);
                    makeQueryRequest(params);
                }, 'it', 'en');
            }
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
            //startRecognition();
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
