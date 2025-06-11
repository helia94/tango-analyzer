import React from 'react'

const BeatsList = ({ beats = [], currentTime, onBeatClick }) => {
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const getCurrentBeatIndex = () => {
    for (let i = 0; i < beats.length; i++) {
      if (beats[i].time > currentTime) {
        return i - 1
      }
    }
    return beats.length - 1
  }

  const currentBeatIndex = getCurrentBeatIndex()

  return (
    <div className="space-y-2">
      <h3 className="text-lg font-semibold text-gray-900">Beat Timeline</h3>
      <div className="max-h-96 overflow-y-auto border border-gray-200 rounded-lg">
        <div className="grid grid-cols-1 gap-1 p-2">
          {beats.map((beat, index) => {
            const isActive = index === currentBeatIndex
            const isPast = beat.time < currentTime
            const confidencePercent = Math.round((beat.confidence || 0) * 100)
            
            return (
              <div
                key={index}
                className={`flex items-center justify-between p-3 rounded-lg cursor-pointer transition-all duration-200 ${
                  isActive 
                    ? 'bg-blue-100 border-2 border-blue-500 shadow-md' 
                    : isPast 
                    ? 'bg-gray-50 hover:bg-gray-100' 
                    : 'bg-white hover:bg-gray-50 border border-gray-200'
                }`}
                onClick={() => onBeatClick(beat.time)}
              >
                <div className="flex items-center space-x-3">
                  <div className={`w-3 h-3 rounded-full ${
                    isActive ? 'bg-blue-500 animate-pulse' : 
                    isPast ? 'bg-gray-400' : 'bg-gray-300'
                  }`} />
                  <div>
                    <span className={`font-medium ${
                      isActive ? 'text-blue-900' : 'text-gray-900'
                    }`}>
                      Beat {index + 1}
                    </span>
                    <div className="text-sm text-gray-500">
                      {formatTime(beat.time)}
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2">
                  <div className="text-right">
                    <div className={`text-sm font-medium ${
                      isActive ? 'text-blue-700' : 'text-gray-700'
                    }`}>
                      {confidencePercent}%
                    </div>
                    <div className="text-xs text-gray-500">confidence</div>
                  </div>
                  
                  {/* Confidence bar */}
                  <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                    <div 
                      className={`h-full transition-all duration-300 ${
                        isActive ? 'bg-blue-500' : 
                        confidencePercent > 70 ? 'bg-green-500' :
                        confidencePercent > 40 ? 'bg-yellow-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${confidencePercent}%` }}
                    />
                  </div>
                </div>
              </div>
            )
          })}
        </div>
        
        {beats.length === 0 && (
          <div className="p-8 text-center text-gray-500">
            <div className="text-lg mb-2">No beats detected</div>
            <div className="text-sm">Upload and analyze a music file to see beats</div>
          </div>
        )}
      </div>
      
      {beats.length > 0 && (
        <div className="text-sm text-gray-600 bg-gray-50 p-3 rounded-lg">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <span className="font-medium">Total Beats:</span> {beats.length}
            </div>
            <div>
              <span className="font-medium">Current Beat:</span> {currentBeatIndex + 1}
            </div>
            <div>
              <span className="font-medium">Average Confidence:</span>{' '}
              {Math.round(beats.reduce((sum, beat) => sum + (beat.confidence || 0), 0) / beats.length * 100)}%
            </div>
            <div>
              <span className="font-medium">Next Beat:</span>{' '}
              {currentBeatIndex < beats.length - 1 
                ? `in ${(beats[currentBeatIndex + 1].time - currentTime).toFixed(1)}s`
                : 'End of song'
              }
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default BeatsList

