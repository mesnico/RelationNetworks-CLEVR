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
.controller('clevrImagesController', ['$scope', '$http', '$timeout', function($scope, $http, $timeout){
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

    //adjustAffixWidth();

    /*$http.get('http://datone.isti.cnr.it/adversarials/adv_scores.json').then(function(response){
        thiz.data = $filter('orderBy')(response.data, 'knn_score', true);
        thiz.loadedData = thiz.data.slice(0, 20);
    });*/
    $scope.changeQueryDivStyle = {'display': 'none'};
    $scope.chooseQueryDivStyle = {'display': 'block'};
    $scope.template = 'templateChooseImage.html';
                            
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
    load();

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
        var init_off = 501;
        if (queryChoice>=0) {
            return 750 + 0.8*(off.top - init_off);
        } else {
            var off = getAbsOffset($( ".samples-container" )[ 0 ]);
            return 510 + 0.8*(off.top - init_off);
        }
    }

    $scope.queryChoice = function() {
        $scope.template = 'templateChooseImage.html';
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
                thiz.questions = response.data;
                questionChoice = response.data[0].id;  //the first question on this image      
            
            /*head = 0;
            nextStop = maxLoadedImages;
            load();
            thiz.loadMoreDisabled = false;*/
            
        });
        
    };

    $scope.chosenQuestion = function(id) {
        questionChoice = id;
        $http({
            url: '/requests/answer',
            method: 'GET',
            params: {qstid: id}}). then(function(response){
                thiz.answer = response.data[0];
                $scope.answerDivStyle = {'display': 'block'};  
                if (response.data[0] == response.data[1]) {
                    $scope.answerDivStyle = {'color': 'green'}; 
                } else {
                    $scope.answerDivStyle = {'color': 'red'};
                }
            $timeout(function(){
                $scope.answerDivStyle = {'display': 'none'}; 
            }, 2000);
            /*head = 0;
            nextStop = maxLoadedImages;
            load();
            thiz.loadMoreDisabled = false;*/
            
        });  
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
});
