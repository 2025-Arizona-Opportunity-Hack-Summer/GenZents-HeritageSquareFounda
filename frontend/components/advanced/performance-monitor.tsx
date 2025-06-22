"use client"

import { Input } from "@/components/ui/input"

import { Label } from "@/components/ui/label"

import * as React from "react"
import {
  Activity,
  Cpu,
  HardDrive,
  Wifi,
  Zap,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  CheckCircle,
  Clock,
} from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from "recharts"
import type { PerformanceMetric, SystemHealth } from "@/lib/types/ui"

// Remove duplicate interfaces as they now come from centralized types

const generateMockData = (points: number) => {
  return Array.from({ length: points }, (_, i) => ({
    timestamp: new Date(Date.now() - (points - i) * 60000),
    value: Math.random() * 100 + Math.sin(i * 0.1) * 20 + 50,
  }))
}

const mockMetrics: PerformanceMetric[] = [
  {
    id: "response-time",
    name: "Response Time",
    value: 245,
    unit: "ms",
    status: "good",
    trend: "down",
    history: generateMockData(30),
  },
  {
    id: "throughput",
    name: "Throughput",
    value: 1250,
    unit: "req/min",
    status: "good",
    trend: "up",
    history: generateMockData(30),
  },
  {
    id: "error-rate",
    name: "Error Rate",
    value: 0.8,
    unit: "%",
    status: "warning",
    trend: "up",
    history: generateMockData(30),
  },
  {
    id: "cpu-usage",
    name: "CPU Usage",
    value: 68,
    unit: "%",
    status: "warning",
    trend: "stable",
    history: generateMockData(30),
  },
  {
    id: "memory-usage",
    name: "Memory Usage",
    value: 72,
    unit: "%",
    status: "good",
    trend: "stable",
    history: generateMockData(30),
  },
  {
    id: "network-latency",
    name: "Network Latency",
    value: 45,
    unit: "ms",
    status: "good",
    trend: "down",
    history: generateMockData(30),
  },
]

const mockSystemHealth: SystemHealth = {
  overall: 85,
  cpu: 68,
  memory: 72,
  network: 95,
  storage: 45,
}

export function PerformanceMonitor() {
  const [metrics, setMetrics] = React.useState<PerformanceMetric[]>(mockMetrics)
  const [systemHealth, setSystemHealth] = React.useState<SystemHealth>(mockSystemHealth)
  const [selectedMetric, setSelectedMetric] = React.useState<PerformanceMetric>(metrics[0])

  // Simulate real-time updates
  React.useEffect(() => {
    const interval = setInterval(() => {
      setMetrics((prev) =>
        prev.map((metric) => ({
          ...metric,
          value: Math.max(0, metric.value + (Math.random() - 0.5) * 10),
          history: [
            ...metric.history.slice(1),
            {
              timestamp: new Date(),
              value: Math.max(0, metric.value + (Math.random() - 0.5) * 10),
            },
          ],
        }))
      )

      setSystemHealth((prev) => ({
        ...prev,
        overall: Math.max(0, Math.min(100, prev.overall + (Math.random() - 0.5) * 5)),
        cpu: Math.max(0, Math.min(100, prev.cpu + (Math.random() - 0.5) * 10)),
        memory: Math.max(0, Math.min(100, prev.memory + (Math.random() - 0.5) * 5)),
        network: Math.max(0, Math.min(100, prev.network + (Math.random() - 0.5) * 3)),
        storage: Math.max(0, Math.min(100, prev.storage + (Math.random() - 0.5) * 2)),
      }))
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  const getStatusColor = (status: PerformanceMetric["status"]) => {
    switch (status) {
      case "good":
        return "text-green-600"
      case "warning":
        return "text-yellow-600"
      case "critical":
        return "text-red-600"
      default:
        return "text-gray-600"
    }
  }

  const getStatusIcon = (status: PerformanceMetric["status"]) => {
    switch (status) {
      case "good":
        return CheckCircle
      case "warning":
        return AlertTriangle
      case "critical":
        return AlertTriangle
      default:
        return Activity
    }
  }

  const getTrendIcon = (trend: PerformanceMetric["trend"]) => {
    switch (trend) {
      case "up":
        return TrendingUp
      case "down":
        return TrendingDown
      case "stable":
        return Activity
      default:
        return Activity
    }
  }

  const getHealthColor = (value: number) => {
    if (value >= 80) return "text-green-600"
    if (value >= 60) return "text-yellow-600"
    return "text-red-600"
  }

  const getHealthStatus = (value: number) => {
    if (value >= 80) return "Excellent"
    if (value >= 60) return "Good"
    if (value >= 40) return "Fair"
    return "Poor"
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Activity className="h-5 w-5" />
            Performance Monitor
          </h3>
          <p className="text-sm text-muted-foreground">Real-time system performance and health metrics</p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="flex items-center gap-1">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
            Live
          </Badge>
          <Badge variant="secondary">Last updated: {new Date().toLocaleTimeString()}</Badge>
        </div>
      </div>

      <Tabs defaultValue="overview" className="space-y-4">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
          <TabsTrigger value="system">System Health</TabsTrigger>
          <TabsTrigger value="alerts">Alerts</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4">
          {/* System Health Score */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="h-5 w-5" />
                Overall System Health
              </CardTitle>
              <CardDescription>Current system performance score</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center justify-center">
                <div className="relative w-32 h-32">
                  <svg className="w-32 h-32 transform -rotate-90" viewBox="0 0 120 120">
                    <circle
                      cx="60"
                      cy="60"
                      r="50"
                      stroke="currentColor"
                      strokeWidth="8"
                      fill="none"
                      className="text-muted-foreground/20"
                    />
                    <circle
                      cx="60"
                      cy="60"
                      r="50"
                      stroke="currentColor"
                      strokeWidth="8"
                      fill="none"
                      strokeDasharray={`${(systemHealth.overall / 100) * 314} 314`}
                      className={getHealthColor(systemHealth.overall)}
                    />
                  </svg>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="text-center">
                      <div className={`text-2xl font-bold ${getHealthColor(systemHealth.overall)}`}>
                        {Math.round(systemHealth.overall)}%
                      </div>
                      <div className="text-sm text-muted-foreground">{getHealthStatus(systemHealth.overall)}</div>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Key Metrics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {metrics.slice(0, 6).map((metric) => {
              const StatusIcon = getStatusIcon(metric.status)
              const TrendIcon = getTrendIcon(metric.trend)

              return (
                <Card key={metric.id} className="cursor-pointer hover:shadow-md transition-shadow">
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium text-sm">{metric.name}</h4>
                      <StatusIcon className={`h-4 w-4 ${getStatusColor(metric.status)}`} />
                    </div>

                    <div className="flex items-end justify-between">
                      <div>
                        <div className="text-2xl font-bold">
                          {metric.value.toFixed(metric.unit === "%" ? 1 : 0)}
                          <span className="text-sm font-normal text-muted-foreground ml-1">{metric.unit}</span>
                        </div>
                      </div>
                      <div className="flex items-center gap-1">
                        <TrendIcon className="h-3 w-3 text-muted-foreground" />
                        <span className="text-xs text-muted-foreground capitalize">{metric.trend}</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )
            })}
          </div>

          {/* Performance Chart */}
          <Card>
            <CardHeader>
              <CardTitle>Response Time Trend</CardTitle>
              <CardDescription>Last 30 minutes</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={200}>
                <AreaChart data={selectedMetric.history.map((h) => ({ ...h, timestamp: h.timestamp.getTime() }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timestamp"
                    type="number"
                    scale="time"
                    domain={["dataMin", "dataMax"]}
                    tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                  />
                  <YAxis />
                  <Tooltip
                    labelFormatter={(value) => new Date(value).toLocaleTimeString()}
                    formatter={(value: number) => [`${value.toFixed(1)} ${selectedMetric.unit}`, selectedMetric.name]}
                  />
                  <Area type="monotone" dataKey="value" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.2} />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="metrics" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Metrics List */}
            <Card>
              <CardHeader>
                <CardTitle>Performance Metrics</CardTitle>
                <CardDescription>Current system performance indicators</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {metrics.map((metric) => {
                    const StatusIcon = getStatusIcon(metric.status)
                    const TrendIcon = getTrendIcon(metric.trend)

                    return (
                      <div
                        key={metric.id}
                        className={`p-3 border rounded cursor-pointer transition-colors ${
                          selectedMetric.id === metric.id ? "border-primary bg-primary/5" : ""
                        }`}
                        onClick={() => setSelectedMetric(metric)}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <StatusIcon className={`h-4 w-4 ${getStatusColor(metric.status)}`} />
                            <div>
                              <p className="font-medium text-sm">{metric.name}</p>
                              <p className="text-xs text-muted-foreground">
                                {metric.value.toFixed(metric.unit === "%" ? 1 : 0)} {metric.unit}
                              </p>
                            </div>
                          </div>

                          <div className="flex items-center gap-2">
                            <TrendIcon className="h-3 w-3 text-muted-foreground" />
                            <Badge variant="outline" className={getStatusColor(metric.status)}>
                              {metric.status}
                            </Badge>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </CardContent>
            </Card>

            {/* Selected Metric Chart */}
            <Card>
              <CardHeader>
                <CardTitle>{selectedMetric.name} History</CardTitle>
                <CardDescription>Last 30 data points</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={selectedMetric.history.map((h) => ({ ...h, timestamp: h.timestamp.getTime() }))}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="timestamp"
                      type="number"
                      scale="time"
                      domain={["dataMin", "dataMax"]}
                      tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                    />
                    <YAxis />
                    <Tooltip
                      labelFormatter={(value) => new Date(value).toLocaleTimeString()}
                      formatter={(value: number) => [`${value.toFixed(1)} ${selectedMetric.unit}`, selectedMetric.name]}
                    />
                    <Line
                      type="monotone"
                      dataKey="value"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={{ fill: "#3b82f6", strokeWidth: 2, r: 3 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="system" className="space-y-4">
          {/* System Resources */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <Cpu className="h-8 w-8 text-blue-500" />
                  <div className="flex-1">
                    <p className="text-sm font-medium">CPU Usage</p>
                    <p className="text-2xl font-bold">{Math.round(systemHealth.cpu)}%</p>
                    <Progress value={systemHealth.cpu} className="mt-2" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <HardDrive className="h-8 w-8 text-green-500" />
                  <div className="flex-1">
                    <p className="text-sm font-medium">Memory Usage</p>
                    <p className="text-2xl font-bold">{Math.round(systemHealth.memory)}%</p>
                    <Progress value={systemHealth.memory} className="mt-2" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <Wifi className="h-8 w-8 text-purple-500" />
                  <div className="flex-1">
                    <p className="text-sm font-medium">Network</p>
                    <p className="text-2xl font-bold">{Math.round(systemHealth.network)}%</p>
                    <Progress value={systemHealth.network} className="mt-2" />
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center gap-3">
                  <HardDrive className="h-8 w-8 text-orange-500" />
                  <div className="flex-1">
                    <p className="text-sm font-medium">Storage</p>
                    <p className="text-2xl font-bold">{Math.round(systemHealth.storage)}%</p>
                    <Progress value={systemHealth.storage} className="mt-2" />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* System Information */}
          <Card>
            <CardHeader>
              <CardTitle>System Information</CardTitle>
              <CardDescription>Current system status and configuration</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <Label className="text-sm font-medium">Operating System</Label>
                    <p className="text-sm text-muted-foreground">Linux Ubuntu 22.04 LTS</p>
                  </div>
                  <div>
                    <Label className="text-sm font-medium">Runtime</Label>
                    <p className="text-sm text-muted-foreground">Node.js 18.17.0</p>
                  </div>
                  <div>
                    <Label className="text-sm font-medium">Memory</Label>
                    <p className="text-sm text-muted-foreground">16 GB DDR4</p>
                  </div>
                  <div>
                    <Label className="text-sm font-medium">Storage</Label>
                    <p className="text-sm text-muted-foreground">512 GB SSD</p>
                  </div>
                </div>
                <div className="space-y-4">
                  <div>
                    <Label className="text-sm font-medium">Uptime</Label>
                    <p className="text-sm text-muted-foreground">7 days, 14 hours, 32 minutes</p>
                  </div>
                  <div>
                    <Label className="text-sm font-medium">Load Average</Label>
                    <p className="text-sm text-muted-foreground">0.85, 0.92, 1.05</p>
                  </div>
                  <div>
                    <Label className="text-sm font-medium">Active Connections</Label>
                    <p className="text-sm text-muted-foreground">1,247 connections</p>
                  </div>
                  <div>
                    <Label className="text-sm font-medium">Cache Hit Rate</Label>
                    <p className="text-sm text-muted-foreground">94.2%</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="alerts" className="space-y-4">
          {/* Active Alerts */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertTriangle className="h-5 w-5 text-yellow-500" />
                Active Alerts
              </CardTitle>
              <CardDescription>Current system alerts and warnings</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex items-center gap-3 p-3 border rounded bg-yellow-50">
                  <AlertTriangle className="h-4 w-4 text-yellow-600" />
                  <div className="flex-1">
                    <p className="font-medium text-sm">High CPU Usage</p>
                    <p className="text-xs text-muted-foreground">CPU usage has been above 80% for 5 minutes</p>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    <Clock className="h-3 w-3 inline mr-1" />5 min ago
                  </div>
                </div>

                <div className="flex items-center gap-3 p-3 border rounded bg-red-50">
                  <AlertTriangle className="h-4 w-4 text-red-600" />
                  <div className="flex-1">
                    <p className="font-medium text-sm">Memory Leak Detected</p>
                    <p className="text-xs text-muted-foreground">Memory usage increasing steadily over time</p>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    <Clock className="h-3 w-3 inline mr-1" />
                    15 min ago
                  </div>
                </div>

                <div className="flex items-center gap-3 p-3 border rounded bg-blue-50">
                  <CheckCircle className="h-4 w-4 text-blue-600" />
                  <div className="flex-1">
                    <p className="font-medium text-sm">Database Connection Restored</p>
                    <p className="text-xs text-muted-foreground">Connection to primary database is now stable</p>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    <Clock className="h-3 w-3 inline mr-1" />1 hour ago
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Alert Configuration */}
          <Card>
            <CardHeader>
              <CardTitle>Alert Thresholds</CardTitle>
              <CardDescription>Configure when alerts should be triggered</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label>CPU Usage Warning (%)</Label>
                    <Input type="number" defaultValue="70" />
                  </div>
                  <div className="space-y-2">
                    <Label>CPU Usage Critical (%)</Label>
                    <Input type="number" defaultValue="90" />
                  </div>
                  <div className="space-y-2">
                    <Label>Memory Usage Warning (%)</Label>
                    <Input type="number" defaultValue="80" />
                  </div>
                  <div className="space-y-2">
                    <Label>Memory Usage Critical (%)</Label>
                    <Input type="number" defaultValue="95" />
                  </div>
                  <div className="space-y-2">
                    <Label>Response Time Warning (ms)</Label>
                    <Input type="number" defaultValue="500" />
                  </div>
                  <div className="space-y-2">
                    <Label>Response Time Critical (ms)</Label>
                    <Input type="number" defaultValue="1000" />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
