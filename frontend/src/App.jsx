import { useState, useRef } from 'react'
import { Upload, Music, Activity, BarChart3, Download, Play, Pause, Volume2, Eye, Waves, Filter, Sliders } from 'lucide-react'
import { Button } from '@/components/ui/button.jsx'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card.jsx'
import { Progress } from '@/components/ui/progress.jsx'
import { Badge } from '@/components/ui/badge.jsx'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs.jsx'
import { Switch } from '@/components/ui/switch.jsx'
import { Label } from '@/components/ui/label.jsx'
import FilteredMusicVisualizer from '@/components/FilteredMusicVisualizer.jsx'
import FilteredBeatsList from '@/components/FilteredBeatsList.jsx'
import './App.css'

const API_BASE_URL = 'http://localhost:5001/api/music'

function App() {
  const [file, setFile] = useState(null)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [analysisResults, setAnalysisResults] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [error, setError] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const [audioUrl, setAudioUrl] = useState(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [visualizationMode, setVisualizationMode] = useState('beats') // 'beats' or 'melody'
  
  const fileInputRef = useRef(null)
  const audioRef = useRef(null)

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0])
    }
  }

  const handleFileSelect = (selectedFile) => {
    const allowedTypes = ['audio/mp3', 'audio/wav', 'audio/flac', 'audio/m4a', 'audio/aac', 'audio/ogg']
    const maxSize = 50 * 1024 * 1024 // 50MB
    
    if (!allowedTypes.includes(selectedFile.type) && !selectedFile.name.match(/\.(mp3|wav|flac|m4a|aac|ogg)$/i)) {
      setError('Please select a valid audio file (MP3, WAV, FLAC, M4A, AAC, OGG)')
      return
    }
    
    if (selectedFile.size > maxSize) {
      setError('File size must be less than 50MB')
      return
    }
    
    setFile(selectedFile)
    setError(null)
    setAnalysisResults(null)
    setCurrentTime(0)
    setDuration(0)
    
    // Create audio URL for playback
    const url = URL.createObjectURL(selectedFile)
    setAudioUrl(url)
  }

  const uploadAndAnalyze = async () => {
    if (!file) return
    
    setIsAnalyzing(true)
    setError(null)
    setUploadProgress(0)
    
    try {
      // Test backend connection first
      console.log('Testing backend connection...')
      const healthResponse = await fetch(`${API_BASE_URL}/health`)
      if (!healthResponse.ok) {
        throw new Error(`Backend not responding. Status: ${healthResponse.status}`)
      }
      console.log('Backend connection successful')
      
      // Upload file
      console.log('Uploading file...')
      const formData = new FormData()
      formData.append('file', file)
      
      const uploadResponse = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      })
      
      if (!uploadResponse.ok) {
        const errorText = await uploadResponse.text()
        throw new Error(`Upload failed: ${uploadResponse.status} - ${errorText}`)
      }
      
      const uploadResult = await uploadResponse.json()
      console.log('Upload successful:', uploadResult)
      setUploadProgress(50)
      
      // Analyze file
      console.log('Starting analysis...')
      const analyzeResponse = await fetch(`${API_BASE_URL}/analyze/${uploadResult.upload_id}`, {
        method: 'POST',
      })
      
      if (!analyzeResponse.ok) {
        const errorText = await analyzeResponse.text()
        throw new Error(`Analysis failed: ${analyzeResponse.status} - ${errorText}`)
      }
      
      const analysisResult = await analyzeResponse.json()
      console.log('Analysis successful:', analysisResult)
      setAnalysisResults(analysisResult)
      setUploadProgress(100)
      
    } catch (err) {
      console.error('Error:', err)
      setError(err.message || 'An error occurred during analysis')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const togglePlayback = () => {
    if (!audioRef.current) return
    
    if (isPlaying) {
      audioRef.current.pause()
    } else {
      audioRef.current.play()
    }
    setIsPlaying(!isPlaying)
  }

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime)
    }
  }

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration)
    }
  }

  const handleSeek = (time) => {
    if (audioRef.current) {
      audioRef.current.currentTime = time
      setCurrentTime(time)
    }
  }

  const handleBeatClick = (time) => {
    handleSeek(time)
  }

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const exportResults = (format) => {
    if (!analysisResults) return
    
    let data, filename, mimeType
    
    if (format === 'json') {
      data = JSON.stringify(analysisResults, null, 2)
      filename = 'tango_analysis.json'
      mimeType = 'application/json'
    } else if (format === 'csv') {
      // Export beats as CSV
      const beats = analysisResults.beat_analysis.beats
      const csvContent = 'Time,Confidence,Strength\n' + 
        beats.map(beat => `${beat.time},${beat.confidence},${beat.strength}`).join('\n')
      data = csvContent
      filename = 'tango_beats.csv'
      mimeType = 'text/csv'
    }
    
    const blob = new Blob([data], { type: mimeType })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-slate-900 dark:text-slate-100 mb-2">
            Argentine Tango Music Analyzer
          </h1>
          <p className="text-lg text-slate-600 dark:text-slate-400">
            Advanced beat filtering with confidence and strength thresholds
          </p>
          <div className="flex items-center justify-center gap-2 mt-2">
            <Filter className="h-5 w-5 text-blue-500" />
            <Badge variant="secondary" className="text-sm">
              Filter beats by confidence & strength
            </Badge>
            <Sliders className="h-5 w-5 text-green-500" />
            <Badge variant="secondary" className="text-sm">
              Adjustable thresholds
            </Badge>
          </div>
        </div>

        {/* File Upload Section */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Upload className="h-5 w-5" />
              Upload Tango Music
            </CardTitle>
            <CardDescription>
              Upload your Argentine tango music file for detailed analysis with filtering (MP3, WAV, FLAC, M4A, AAC, OGG)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                dragActive 
                  ? 'border-primary bg-primary/5' 
                  : 'border-slate-300 dark:border-slate-600 hover:border-primary/50'
              }`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
            >
              {file ? (
                <div className="space-y-4">
                  <Music className="h-12 w-12 mx-auto text-primary" />
                  <div>
                    <p className="font-medium">{file.name}</p>
                    <p className="text-sm text-slate-500">
                      {(file.size / (1024 * 1024)).toFixed(2)} MB
                    </p>
                  </div>
                  
                  <div className="flex gap-2 justify-center">
                    <Button onClick={uploadAndAnalyze} disabled={isAnalyzing}>
                      {isAnalyzing ? 'Analyzing...' : 'Analyze Music'}
                    </Button>
                    <Button 
                      variant="outline" 
                      onClick={() => fileInputRef.current?.click()}
                    >
                      Choose Different File
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <Upload className="h-12 w-12 mx-auto text-slate-400" />
                  <div>
                    <p className="text-lg font-medium">Drop your tango music file here</p>
                    <p className="text-slate-500">or click to browse</p>
                  </div>
                  <Button onClick={() => fileInputRef.current?.click()}>
                    Select File
                  </Button>
                </div>
              )}
              
              <input
                ref={fileInputRef}
                type="file"
                className="hidden"
                accept=".mp3,.wav,.flac,.m4a,.aac,.ogg"
                onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
              />
            </div>
            
            {isAnalyzing && (
              <div className="mt-4 space-y-2">
                <Progress value={uploadProgress} />
                <p className="text-sm text-center text-slate-600">
                  {uploadProgress < 50 ? 'Uploading...' : 'Analyzing...'}
                </p>
              </div>
            )}
            
            {error && (
              <div className="mt-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
                <p className="text-red-700 dark:text-red-400 text-sm">{error}</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Music Player and Filtered Visualization */}
        {audioUrl && (
          <Card className="mb-8">
            <CardHeader>
              <div className="flex justify-between items-center">
                <CardTitle className="flex items-center gap-2">
                  <Volume2 className="h-5 w-5" />
                  Filtered Music Visualization
                </CardTitle>
                
                {analysisResults && (
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-2">
                      <Eye className="h-4 w-4" />
                      <Label htmlFor="viz-mode" className="text-sm">Beats</Label>
                      <Switch
                        id="viz-mode"
                        checked={visualizationMode === 'melody'}
                        onCheckedChange={(checked) => setVisualizationMode(checked ? 'melody' : 'beats')}
                      />
                      <Label htmlFor="viz-mode" className="text-sm">Melody</Label>
                      <Waves className="h-4 w-4" />
                    </div>
                  </div>
                )}
              </div>
              <CardDescription>
                Filter beats by confidence and strength thresholds. Only beats above your thresholds will be shown.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <audio
                ref={audioRef}
                src={audioUrl}
                onTimeUpdate={handleTimeUpdate}
                onLoadedMetadata={handleLoadedMetadata}
                onEnded={() => setIsPlaying(false)}
              />
              
              {/* Player Controls */}
              <div className="flex items-center justify-center gap-4">
                <Button
                  variant="outline"
                  size="lg"
                  onClick={togglePlayback}
                  className="flex items-center gap-2"
                >
                  {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
                  {isPlaying ? 'Pause' : 'Play'}
                </Button>
                
                <div className="text-sm text-gray-600">
                  {formatTime(currentTime)} / {formatTime(duration)}
                </div>
              </div>

              {/* Filtered Music Visualization */}
              {duration > 0 && (
                <FilteredMusicVisualizer
                  audioRef={audioRef}
                  duration={duration}
                  currentTime={currentTime}
                  beats={analysisResults?.beat_analysis?.beats || []}
                  melodySegments={analysisResults?.melody_analysis?.segments || []}
                  tangoSections={analysisResults?.section_analysis?.sections || []}
                  visualizationMode={visualizationMode}
                  onSeek={handleSeek}
                />
              )}
            </CardContent>
          </Card>
        )}

        {/* Analysis Results */}
        {analysisResults && (
          <div className="space-y-6">
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium flex items-center gap-2">
                    <Activity className="h-4 w-4" />
                    Beat Analysis
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {analysisResults.beat_analysis.bpm.toFixed(1)} BPM
                  </div>
                  <p className="text-xs text-slate-500">
                    {analysisResults.beat_analysis.beats.length} beats detected
                  </p>
                  <Badge variant="secondary" className="mt-2">
                    {analysisResults.beat_analysis.method}
                  </Badge>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium flex items-center gap-2">
                    <Music className="h-4 w-4" />
                    Melody Style
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-sm">Legato</span>
                      <span className="text-sm font-medium">
                        {analysisResults.melody_analysis.statistics.legato_percentage.toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-sm">Staccato</span>
                      <span className="text-sm font-medium">
                        {analysisResults.melody_analysis.statistics.staccato_percentage.toFixed(1)}%
                      </span>
                    </div>
                  </div>
                  <p className="text-xs text-slate-500 mt-2">
                    {analysisResults.melody_analysis.statistics.total_segments} segments
                  </p>
                </CardContent>
              </Card>
              
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium flex items-center gap-2">
                    <BarChart3 className="h-4 w-4" />
                    Tango Structure
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">
                    {analysisResults.section_analysis.sections.length} Sections
                  </div>
                  <p className="text-xs text-slate-500">
                    {analysisResults.time_signature} time signature
                  </p>
                  <div className="flex gap-1 mt-2">
                    {analysisResults.section_analysis.sections.map((section, idx) => (
                      <Badge key={idx} variant="outline" className="text-xs">
                        {section.type}
                      </Badge>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Detailed Results */}
            <Card>
              <CardHeader>
                <div className="flex justify-between items-center">
                  <CardTitle>Detailed Analysis with Filtering</CardTitle>
                  <div className="flex gap-2">
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => exportResults('csv')}
                      className="flex items-center gap-2"
                    >
                      <Download className="h-4 w-4" />
                      Export CSV
                    </Button>
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => exportResults('json')}
                      className="flex items-center gap-2"
                    >
                      <Download className="h-4 w-4" />
                      Export JSON
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <Tabs defaultValue="beats" className="w-full">
                  <TabsList className="grid w-full grid-cols-3">
                    <TabsTrigger value="beats">Filtered Beat Detection</TabsTrigger>
                    <TabsTrigger value="melody">Melody Analysis</TabsTrigger>
                    <TabsTrigger value="structure">Tango Structure</TabsTrigger>
                  </TabsList>
                  
                  <TabsContent value="beats" className="space-y-4">
                    <FilteredBeatsList 
                      beats={analysisResults.beat_analysis.beats}
                      currentTime={currentTime}
                      onBeatClick={handleBeatClick}
                    />
                  </TabsContent>
                  
                  <TabsContent value="melody" className="space-y-4">
                    <div className="space-y-2">
                      <h4 className="font-medium">Melody Segments</h4>
                      <div className="max-h-64 overflow-y-auto space-y-1">
                        {analysisResults.melody_analysis.segments.slice(0, 15).map((segment, idx) => (
                          <div key={idx} className="flex justify-between items-center p-2 bg-slate-50 dark:bg-slate-800 rounded">
                            <div>
                              <span className="text-sm font-medium capitalize">{segment.type}</span>
                              <p className="text-xs text-slate-500">
                                {segment.start.toFixed(2)}s - {segment.end.toFixed(2)}s
                              </p>
                            </div>
                            <Badge 
                              variant={segment.type === 'legato' ? 'default' : 'secondary'}
                              className="text-xs"
                            >
                              {(segment.confidence * 100).toFixed(0)}%
                            </Badge>
                          </div>
                        ))}
                        {analysisResults.melody_analysis.segments.length > 15 && (
                          <p className="text-xs text-slate-500 text-center">
                            ... and {analysisResults.melody_analysis.segments.length - 15} more segments
                          </p>
                        )}
                      </div>
                    </div>
                  </TabsContent>
                  
                  <TabsContent value="structure" className="space-y-4">
                    <div className="space-y-4">
                      <div>
                        <h4 className="font-medium mb-2">Sections</h4>
                        <div className="space-y-2">
                          {analysisResults.section_analysis.sections.map((section, idx) => (
                            <div key={idx} className="flex justify-between items-center p-3 bg-slate-50 dark:bg-slate-800 rounded">
                              <div>
                                <span className="font-medium">Section {section.type}</span>
                                <p className="text-sm text-slate-500 capitalize">
                                  {section.description.replace(/_/g, ' ')}
                                </p>
                              </div>
                              <span className="text-sm">
                                {section.start.toFixed(1)}s - {section.end.toFixed(1)}s
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </TabsContent>
                </Tabs>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  )
}

export default App

